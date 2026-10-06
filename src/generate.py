"""Citation-constrained generation through OpenRouter.

Contract (docs/design.md): the answer may use only the provided chunks and must cite
them as ``[cite: <chunk_id>]``. A citation that is not in the retrieved set is a hard
error. Weak retrieval or a model ``NOT_IN_SOURCES`` reply yields a refusal, never an
answer from model priors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.budget import SpendLedger
from src.retrieval import Hit
from src.trace import Tracer

NOT_IN_SOURCES = "NOT_IN_SOURCES"
REFUSAL_TEXT = "Not in my sources. I can only answer from the documents I was given."
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# USD per 1M tokens (input, output). Estimates used only when the API reports no cost.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
}
DEFAULT_PRICE = (3.0, 15.0)  # conservative for unknown models

_CITE = re.compile(r"\[cite:\s*([^\]]+)\]", re.I)

SYSTEM_PROMPT = f"""You are a professional-kitchen reference assistant. Answer ONLY from the \
numbered source passages in the user message. Never use outside knowledge.

Rules:
1. Cite every claim as [cite: <chunk_id>] using the exact chunk_id shown. Several ids: \
[cite: id1, id2]. Only cite ids that were provided.
2. If the passages do not contain the answer, reply with exactly {NOT_IN_SOURCES} and nothing else.
3. Sources of type fda and usda are current US government food-safety guidance. If a \
gutenberg source (a 1800s-1900s cookbook) disagrees with fda or usda on safety, temperature \
or time, follow fda/usda, say the older text differs, and cite both.
4. Be concise and give exact temperatures, times and ratios when the sources do."""


class CitationError(RuntimeError):
    """Base class for citation contract violations."""


class InvalidCitationError(CitationError):
    """The answer cited a chunk_id that was not in the retrieved set."""


class MissingCitationError(CitationError):
    """The answer made claims without citing any chunk."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    model: str


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], model: str) -> LLMResult: ...


@dataclass
class Answer:
    text: str
    citations: list[str]
    refused: bool
    reason: str | None
    hits: list[Hit]
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES_PER_MTOK.get(model, DEFAULT_PRICE)
    return prompt_tokens * p_in / 1e6 + completion_tokens * p_out / 1e6


class OpenRouterClient:
    """``openai`` SDK pointed at OpenRouter. Checks and records the spend ledger."""

    def __init__(
        self,
        api_key: str | None = None,
        ledger: SpendLedger | None = None,
        sdk: Any | None = None,
        max_tokens: int = 600,
        label: str = "generate",
    ) -> None:
        if sdk is None:
            from openai import OpenAI

            sdk = OpenAI(base_url=OPENROUTER_URL, api_key=api_key)
        self._sdk: Any = sdk
        self._ledger = ledger
        self._max_tokens = max_tokens
        self._label = label

    def complete(self, messages: list[dict[str, str]], model: str) -> LLMResult:
        if self._ledger:
            self._ledger.check()
        resp = self._sdk.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=self._max_tokens,
            extra_body={"usage": {"include": True}},
        )
        usage = resp.usage
        p_tok, c_tok = int(usage.prompt_tokens), int(usage.completion_tokens)
        reported = getattr(usage, "cost", None)
        cost = float(reported) if reported is not None else estimate_cost(model, p_tok, c_tok)
        if self._ledger:
            self._ledger.record(self._label, model, p_tok, c_tok, cost)
        return LLMResult(resp.choices[0].message.content or "", p_tok, c_tok, cost, model)


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[chunk_id: {h.chunk_id}] (source_type: {h.source_type}; title: {h.title})\n{h.text}"
        for h in hits
    )
    user = f"Source passages:\n\n{passages}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_citations(text: str) -> list[str]:
    """Ordered, de-duplicated chunk_ids from ``[cite: ...]`` markers."""
    seen: dict[str, None] = {}
    for group in _CITE.findall(text):
        for part in re.split(r"[,;]", group):
            if part.strip():
                seen.setdefault(part.strip(), None)
    return list(seen)


def is_weak(hits: list[Hit], min_rerank_score: float | None) -> bool:
    """Weak when nothing was retrieved, or the best rerank score is below the gate."""
    if not hits:
        return True
    scores = [h.rerank_score for h in hits if h.rerank_score is not None]
    return bool(min_rerank_score is not None and scores and max(scores) < min_rerank_score)


def answer_question(
    question: str,
    hits: list[Hit],
    client: LLMClient,
    model: str,
    min_rerank_score: float | None = 0.0,
    tracer: Tracer | None = None,
) -> Answer:
    """Generate a cited answer, or refuse. Raises ``CitationError`` on contract violations."""
    from contextlib import nullcontext

    span: Any = (
        tracer.span("answer", input={"question": question, "chunk_ids": [h.chunk_id for h in hits]})
        if tracer
        else nullcontext()
    )
    with span as root:

        def finish(ans: Answer) -> Answer:
            if root is not None:
                root.update(
                    output={"refused": ans.refused, "reason": ans.reason, "text": ans.text},
                    cost=ans.cost,
                )
            return ans

        if is_weak(hits, min_rerank_score):
            return finish(Answer(REFUSAL_TEXT, [], True, "weak_retrieval", hits))
        gen_span: Any = (
            tracer.span("generate", kind="generation", input=question) if tracer else nullcontext()
        )
        with gen_span as gen:
            res = client.complete(build_messages(question, hits), model)
            if gen is not None:
                gen.update(
                    output=res.text,
                    model=res.model,
                    usage={
                        "prompt_tokens": res.prompt_tokens,
                        "completion_tokens": res.completion_tokens,
                    },
                    cost=res.cost,
                )

        def make(text: str, cites: list[str], refused: bool, reason: str | None) -> Answer:
            return Answer(
                text,
                cites,
                refused,
                reason,
                hits,
                model=res.model,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                cost=res.cost,
            )

        if res.text.strip().upper().startswith(NOT_IN_SOURCES):
            return finish(make(REFUSAL_TEXT, [], True, "model"))
        cited = extract_citations(res.text)
        allowed = {h.chunk_id for h in hits}
        bad = [c for c in cited if c not in allowed]
        if bad:
            raise InvalidCitationError(f"citations not in retrieved set: {bad}")
        if not cited:
            raise MissingCitationError("answer contains no [cite: chunk_id] markers")
        return finish(make(res.text.strip(), cited, False, None))
