"""Tests for the demo service: API shape, rate limits, daily token cap, no question logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.demo import DailyTokenBudget, SlidingWindowLimiter, create_app
from src.generate import Answer
from src.retrieval import Hit
from src.trace import JsonlTracer

SECRET_Q = "how do I zzqx-secret-braise a shank"


def _hit(cid: str, rr: float | None = 2.0) -> Hit:
    return Hit(cid, f"passage {cid}", "p.txt", "Title", "fda", 0.03, 1, 2, rr)


class FakeEngine:
    def __init__(self, can_generate: bool = True, tokens: int = 1000) -> None:
        self.can_generate = can_generate
        self.tokens = tokens
        self.calls = 0

    def retrieve(self, question: str) -> list[Hit]:
        return [_hit("a-0000-aaaaaaaa"), _hit("b-0001-bbbbbbbb")]

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        self.calls += 1
        return Answer(
            "Keep it hot [cite: a-0000-aaaaaaaa].",
            ["a-0000-aaaaaaaa"],
            False,
            None,
            hits,
            model="m",
            prompt_tokens=self.tokens - 100,
            completion_tokens=100,
            cost=0.001,
        )


def make_client(tmp_path: Path, engine: Any = None, **kwargs: Any) -> tuple[TestClient, Any]:
    engine = engine or FakeEngine()
    tracer = JsonlTracer(tmp_path / "traces", redact_input=True)
    app = create_app(engine, tracer=tracer, **kwargs)
    return TestClient(app), engine


def test_query_returns_answer_citations_and_scores(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    r = client.post("/query", json={"question": "what temp?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False and "[cite: a-0000-aaaaaaaa]" in body["answer"]
    assert body["citations"][0]["chunk_id"] == "a-0000-aaaaaaaa"
    assert body["citations"][0]["text"] == "passage a-0000-aaaaaaaa"
    assert [s["chunk_id"] for s in body["scores"]] == ["a-0000-aaaaaaaa", "b-0001-bbbbbbbb"]
    assert {"rrf", "bm25_rank", "dense_rank", "rerank"} <= set(body["scores"][0])


def test_index_page_served(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "innerHTML" not in r.text  # model output must be inserted as text only


def test_input_validation(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.post("/query", json={"question": "   "}).status_code == 422
    assert client.post("/query", json={"question": "x" * 501}).status_code == 422
    assert client.post("/query", json={}).status_code == 422


def test_rate_limit_per_ip(tmp_path: Path) -> None:
    limiter = SlidingWindowLimiter([(2, 60)])
    client, engine = make_client(tmp_path, limiter=limiter)
    h1 = {"x-forwarded-for": "1.1.1.1"}
    assert client.post("/query", json={"question": "a"}, headers=h1).status_code == 200
    assert client.post("/query", json={"question": "a"}, headers=h1).status_code == 200
    r = client.post("/query", json={"question": "a"}, headers=h1)
    assert r.status_code == 429 and "Retry-After" in r.headers
    assert r.json()["error"] == "rate_limited"
    other = {"x-forwarded-for": "2.2.2.2"}
    assert client.post("/query", json={"question": "a"}, headers=other).status_code == 200
    assert engine.calls == 3


def test_sliding_window_expires() -> None:
    now = [0.0]
    lim = SlidingWindowLimiter([(1, 10)], clock=lambda: now[0])
    assert lim.check("ip") is None
    assert lim.check("ip") is not None
    now[0] = 11.0
    assert lim.check("ip") is None


def test_daily_budget_friendly_response_and_reset(tmp_path: Path) -> None:
    now = [1_000_000.0]
    budget = DailyTokenBudget(cap_tokens=1500, clock=lambda: now[0])
    client, engine = make_client(tmp_path, budget=budget, limiter=SlidingWindowLimiter([(99, 60)]))
    assert client.post("/query", json={"question": "a"}).status_code == 200
    assert client.post("/query", json={"question": "a"}).status_code == 200  # 1000 + 1000 > cap
    r = client.post("/query", json={"question": "a"})
    assert r.status_code == 429 and r.json()["error"] == "demo_budget_reached"
    assert "demo budget" in r.json()["message"].lower()
    assert engine.calls == 2
    now[0] += 86_400
    assert client.post("/query", json={"question": "a"}).status_code == 200


def test_no_generation_without_key_still_shows_retrieval(tmp_path: Path) -> None:
    engine = FakeEngine(can_generate=False)
    engine.generate = lambda q, h: None  # type: ignore[method-assign,assignment,return-value]
    client, _ = make_client(tmp_path, engine=engine)
    body = client.post("/query", json={"question": "a"}).json()
    assert body["answer"] is None and body["generation_enabled"] is False
    assert body["scores"]


def test_question_text_never_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    client, _ = make_client(tmp_path)
    with caplog.at_level(logging.DEBUG):
        client.post("/query", json={"question": SECRET_Q})
    assert SECRET_Q not in caplog.text
    traces = "".join(p.read_text() for p in (tmp_path / "traces").glob("*.jsonl"))
    assert traces and SECRET_Q not in traces


def test_healthz(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.get("/healthz").json() == {"ok": True}
