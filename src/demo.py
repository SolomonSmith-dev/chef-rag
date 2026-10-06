"""Public demo service (FastAPI): POST /query plus a small static UI.

Safety rails for a free public deployment:

- the OpenRouter key comes only from the environment (a Space secret);
- per-IP sliding-window rate limit;
- a daily token cap that returns a friendly "demo budget reached" response;
- question text is never logged: the tracer runs with ``redact_input=True`` and this
  module never writes the question to any log.

Run with ``uvicorn --factory src.demo:build_app``.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.generate import Answer, CitationError
from src.retrieval import Hit
from src.trace import Tracer, make_tracer

STATIC = Path(__file__).with_name("static")
BUDGET_MESSAGE = (
    "Demo budget reached for today. The free demo has a daily token cap; "
    "please try again tomorrow (UTC), or run it locally from the repo."
)


class Engine(Protocol):
    def retrieve(self, question: str) -> list[Hit]: ...

    def generate(self, question: str, hits: list[Hit]) -> Answer | None:
        """None when generation is not configured (no API key)."""
        ...


class SlidingWindowLimiter:
    """In-memory per-key limiter. ``limits`` is a list of (max_requests, window_seconds)."""

    def __init__(
        self,
        limits: list[tuple[int, int]],
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Record a request. Returns seconds to wait if limited, else None."""
        now = self._clock()
        longest = max(w for _, w in self.limits)
        with self._lock:
            if len(self._hits) > self._max_keys:
                self._hits.clear()
            q = self._hits[key]
            while q and now - q[0] > longest:
                q.popleft()
            for max_n, window in self.limits:
                recent = [t for t in q if now - t <= window]
                if len(recent) >= max_n:
                    return max(1, int(window - (now - recent[0])) + 1)
            q.append(now)
            return None


class DailyTokenBudget:
    """Token cap that resets at each UTC day boundary."""

    def __init__(self, cap_tokens: int, clock: Callable[[], float] = time.time) -> None:
        self.cap = cap_tokens
        self._clock = clock
        self._day = self._today()
        self.used = 0
        self._lock = threading.Lock()

    def _today(self) -> int:
        return int(self._clock() // 86_400)

    def _roll(self) -> None:
        if self._today() != self._day:
            self._day, self.used = self._today(), 0

    def exhausted(self) -> bool:
        with self._lock:
            self._roll()
            return self.used >= self.cap

    def add(self, tokens: int) -> None:
        with self._lock:
            self._roll()
            self.used += tokens


class QueryRequest(BaseModel):
    question: str = Field(max_length=500)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


def client_ip(request: Request) -> str:
    """Rightmost X-Forwarded-For entry (added by the platform proxy), else the socket peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _score_row(h: Hit) -> dict[str, Any]:
    return {
        "chunk_id": h.chunk_id,
        "title": h.title,
        "source_type": h.source_type,
        "rrf": h.score,
        "bm25_rank": h.bm25_rank,
        "dense_rank": h.dense_rank,
        "rerank": h.rerank_score,
    }


def create_app(
    engine: Engine,
    tracer: Tracer,
    limiter: SlidingWindowLimiter | None = None,
    budget: DailyTokenBudget | None = None,
) -> FastAPI:
    limiter = limiter or SlidingWindowLimiter([(5, 60), (40, 86_400)])
    budget = budget or DailyTokenBudget(200_000)
    app = FastAPI(title="chef-rag demo", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/query")
    def query(req: QueryRequest, request: Request) -> JSONResponse:
        wait = limiter.check(client_ip(request))
        if wait is not None:
            return JSONResponse(
                {"error": "rate_limited", "message": f"Too many requests. Retry in {wait}s."},
                status_code=429,
                headers={"Retry-After": str(wait)},
            )
        if budget.exhausted():
            return JSONResponse(
                {"error": "demo_budget_reached", "message": BUDGET_MESSAGE}, status_code=429
            )
        with tracer.span("demo_query") as span:
            hits = engine.retrieve(req.question)
            refused_reason: str | None = None
            try:
                ans = engine.generate(req.question, hits)
            except CitationError:
                ans, refused_reason = None, "citation_error"
            if ans is not None:
                budget.add(ans.prompt_tokens + ans.completion_tokens)
                cost = ans.cost
            else:
                cost = 0.0
            span.update(
                cost=cost, metadata={"n_hits": len(hits), "refused": bool(ans and ans.refused)}
            )
        cited = ans.citations if ans else []
        by_id = {h.chunk_id: h for h in hits}
        body: dict[str, Any] = {
            "answer": ans.text if ans else None,
            "refused": bool(ans.refused) if ans else refused_reason is not None,
            "reason": (ans.reason if ans else refused_reason),
            "generation_enabled": ans is not None or refused_reason is not None,
            "citations": [
                {
                    "chunk_id": cid,
                    "title": by_id[cid].title,
                    "source_type": by_id[cid].source_type,
                    "text": by_id[cid].text,
                }
                for cid in cited
                if cid in by_id
            ],
            "scores": [_score_row(h) for h in hits],
        }
        if refused_reason:
            body["answer"] = "I could not produce a verified, cited answer. Try rephrasing."
        return JSONResponse(body)

    return app


class DemoEngine:
    """Local-backend engine: hybrid retrieval, bge rerank, optional OpenRouter generation."""

    def __init__(
        self, cfg: Any, backend: Any, reranker: Any, client: Any, model: str, tracer: Tracer
    ):
        self.cfg, self.backend, self.reranker = cfg, backend, reranker
        self.client, self.model, self.tracer = client, model, tracer

    def retrieve(self, question: str) -> list[Hit]:
        from src.pipeline import retrieve

        return retrieve(question, self.backend, self.reranker, self.cfg, tracer=self.tracer)[1]

    def generate(self, question: str, hits: list[Hit]) -> Answer | None:
        from src.generate import answer_question

        if self.client is None:
            return None
        return answer_question(
            question, hits, self.client, self.model, self.cfg.min_rerank_score, self.tracer
        )


def build_app() -> FastAPI:
    """Production wiring from environment variables (use with ``uvicorn --factory``)."""
    from src.config import load_retrieval_settings
    from src.generate import OpenRouterClient
    from src.pipeline import build_backend, build_embedder, build_reranker

    cfg = load_retrieval_settings()
    tracer = make_tracer(os.environ, redact_input=True)
    backend = build_backend("local", cfg, build_embedder("minilm", cfg))
    key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenRouterClient(key, ledger=None, max_tokens=500) if key else None
    engine = DemoEngine(
        cfg,
        backend,
        build_reranker("bge", cfg),
        client,
        os.environ.get("DEMO_MODEL", "anthropic/claude-haiku-4.5"),
        tracer,
    )
    limiter = SlidingWindowLimiter(
        [
            (int(os.environ.get("DEMO_RATE_PER_MIN", "5")), 60),
            (int(os.environ.get("DEMO_RATE_PER_DAY", "40")), 86_400),
        ]
    )
    budget = DailyTokenBudget(int(os.environ.get("DEMO_DAILY_TOKEN_CAP", "200000")))
    return create_app(engine, tracer, limiter, budget)
