"""Tests for citation-constrained generation, refusal, OpenRouter client and spend cap."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.budget import BudgetExceeded, SpendLedger
from src.generate import (
    CitationError,
    InvalidCitationError,
    LLMResult,
    MissingCitationError,
    OpenRouterClient,
    answer_question,
    build_messages,
    estimate_cost,
    extract_citations,
)
from src.retrieval import Hit
from src.trace import JsonlTracer


def hit(cid: str, text: str = "text", st: str = "gutenberg", rr: float | None = 1.0) -> Hit:
    return Hit(cid, text, "p.txt", "Title", st, 0.5, rerank_score=rr)


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text, self.calls = text, 0

    def complete(self, messages: list[dict[str, str]], model: str) -> LLMResult:
        self.calls += 1
        return LLMResult(self.text, 100, 20, 0.002, model)


def test_extract_citations_handles_lists_and_dupes() -> None:
    text = "A [cite: a-0001-x]. B [cite: b-0002-y, a-0001-x]."
    assert extract_citations(text) == ["a-0001-x", "b-0002-y"]


def test_answer_with_valid_citations() -> None:
    hits = [hit("a-1"), hit("b-2")]
    ans = answer_question("q?", hits, FakeClient("Yes [cite: a-1]."), "m")
    assert not ans.refused and ans.citations == ["a-1"]
    assert ans.cost == 0.002 and ans.prompt_tokens == 100


def test_invalid_citation_is_hard_error() -> None:
    with pytest.raises(InvalidCitationError, match="zzz"):
        answer_question("q?", [hit("a-1")], FakeClient("Yes [cite: zzz]."), "m")


def test_uncited_answer_is_hard_error() -> None:
    with pytest.raises(MissingCitationError):
        answer_question("q?", [hit("a-1")], FakeClient("Yes, trust me."), "m")
    assert issubclass(MissingCitationError, CitationError)


def test_weak_retrieval_refuses_without_calling_model() -> None:
    client = FakeClient("should not be used [cite: a-1]")
    ans = answer_question("q?", [hit("a-1", rr=-5.0)], client, "m", min_rerank_score=0.0)
    assert ans.refused and ans.reason == "weak_retrieval" and client.calls == 0
    assert "not in my sources" in ans.text.lower()


def test_empty_retrieval_refuses() -> None:
    client = FakeClient("x")
    ans = answer_question("q?", [], client, "m")
    assert ans.refused and client.calls == 0


def test_no_rerank_scores_means_no_gate() -> None:
    ans = answer_question("q?", [hit("a-1", rr=None)], FakeClient("ok [cite: a-1]"), "m")
    assert not ans.refused


def test_model_refusal_sentinel() -> None:
    ans = answer_question("q?", [hit("a-1")], FakeClient("NOT_IN_SOURCES"), "m")
    assert ans.refused and ans.reason == "model" and ans.citations == []


def test_prompt_contains_ids_and_modern_source_rule() -> None:
    msgs = build_messages("what temp?", [hit("a-1", "165F", st="fda")])
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "[cite: " in system and "NOT_IN_SOURCES" in system
    assert "fda" in system.lower() and "usda" in system.lower()
    assert "a-1" in user and "165F" in user and "what temp?" in user


def test_answer_is_traced_with_generation_span(tmp_path: Path) -> None:
    tracer = JsonlTracer(tmp_path)
    answer_question("q?", [hit("a-1")], FakeClient("ok [cite: a-1]"), "m", tracer=tracer)
    recs = [json.loads(ln) for ln in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    names = {r["name"] for r in recs}
    assert {"answer", "generate"} <= names
    gen = next(r for r in recs if r["name"] == "generate")
    assert gen["kind"] == "generation" and gen["cost"] == 0.002


def test_refusal_is_traced_too(tmp_path: Path) -> None:
    tracer = JsonlTracer(tmp_path)
    answer_question("q?", [], FakeClient("x"), "m", tracer=tracer)
    assert any(tmp_path.glob("*.jsonl"))


class FakeOpenAI:
    def __init__(self, cost: float | None) -> None:
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, cost=cost)
        msg = SimpleNamespace(content="hello")
        self.resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.resp


def test_estimate_cost_known_and_unknown_model() -> None:
    assert estimate_cost("anthropic/claude-sonnet-4", 1_000_000, 0) == pytest.approx(3.0)
    assert estimate_cost("unknown/model", 1_000_000, 0) == pytest.approx(3.0)  # conservative


def test_openrouter_client_uses_reported_cost_and_records(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=5.0)
    fake = FakeOpenAI(cost=0.0123)
    client = OpenRouterClient(sdk=fake, ledger=ledger)
    res = client.complete([{"role": "user", "content": "x"}], "anthropic/claude-sonnet-4")
    assert res.text == "hello" and res.cost == 0.0123
    assert fake.calls[0]["extra_body"] == {"usage": {"include": True}}
    assert ledger.total_usd == pytest.approx(0.0123)


def test_openrouter_client_estimates_when_cost_missing(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "s.json", cap_usd=5.0)
    client = OpenRouterClient(sdk=FakeOpenAI(cost=None), ledger=ledger)
    res = client.complete([{"role": "user", "content": "x"}], "anthropic/claude-sonnet-4")
    assert res.cost == pytest.approx(1000 * 3 / 1e6 + 200 * 15 / 1e6)


def test_budget_blocks_call_before_network(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "s.json", cap_usd=0.01)
    ledger.record("seed", "m", 0, 0, 0.02)
    fake = FakeOpenAI(cost=0.001)
    client = OpenRouterClient(sdk=fake, ledger=ledger)
    with pytest.raises(BudgetExceeded):
        client.complete([{"role": "user", "content": "x"}], "m")
    assert fake.calls == []


def test_ledger_persists_and_projects(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    SpendLedger(path, 5.0).record("run", "m", 10, 5, 1.25)
    again = SpendLedger(path, 5.0)
    assert again.total_usd == pytest.approx(1.25)
    again.ensure_room(3.0)
    with pytest.raises(BudgetExceeded):
        again.ensure_room(4.0)
