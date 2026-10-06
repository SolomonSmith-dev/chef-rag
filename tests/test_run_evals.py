"""Tests for the eval harness: metrics, ablation, live scoring (fakes only), output JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals import run_evals as ev
from src.budget import SpendLedger
from src.generate import LLMResult
from src.ingest import WordTokenizer, ingest_directory, load_chunks
from src.rerank import NoopReranker
from src.retrieval import HashingEmbedder, LocalBackend

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def world(tmp_path: Path) -> tuple[list[Any], LocalBackend, list[dict[str, Any]]]:
    ingest_directory(FIXTURES / "corpus", tmp_path, tokenizer=WordTokenizer())
    chunks = load_chunks(tmp_path / "chunks.jsonl")
    backend = LocalBackend.build(chunks, HashingEmbedder(dim=256))
    return chunks, backend, ev.load_golden(FIXTURES / "golden_fixture.jsonl")


def test_recall_capped_and_mrr() -> None:
    rel = {"a", "b", "c"}
    assert ev.recall_at(rel, ["a", "x", "b"], 3) == pytest.approx(2 / 3)
    # capped by n: 10 relevant, top-2 both relevant -> 1.0
    assert ev.recall_at({str(i) for i in range(10)}, ["0", "1"], 2) == 1.0
    assert ev.reciprocal_rank(rel, ["x", "y", "c"]) == pytest.approx(1 / 3)
    assert ev.reciprocal_rank(rel, ["x"]) == 0.0


def test_percentile() -> None:
    assert ev.percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert ev.percentile([5.0], 95) == 5.0
    assert ev.percentile([], 50) == 0.0


def test_resolve_relevant_and_unresolved(world: Any) -> None:
    chunks, _, records = world
    by_id = {r["id"]: r for r in records}
    assert ev.resolve_relevant(by_id["f-001"], chunks)
    assert ev.resolve_relevant(by_id["f-005"], chunks) == set()
    assert ev.resolve_relevant(by_id["f-006"], chunks) == set()


def test_original_source_selector(world: Any) -> None:
    chunks, _, _ = world
    rec = {"sources": ["original"], "evidence": ["fixture"], "answerable": True}
    ids = ev.resolve_relevant(rec, chunks)
    assert ids and all(c.source_type == "original" for c in chunks if c.chunk_id in ids)


def test_ablation_has_four_configs_and_skips_unresolved(world: Any) -> None:
    chunks, backend, records = world
    result = ev.run_ablation(records, chunks, backend, NoopReranker(), k=2)
    assert set(result["configs"]) == {"bm25", "dense", "hybrid", "hybrid_rerank"}
    assert result["n_questions"] == 4  # f-001..f-004; f-005 unanswerable, f-006 unresolved
    assert result["unresolved"] == ["f-006"]
    for metrics in result["configs"].values():
        assert set(metrics) >= {"recall@10", "recall@k", "mrr"}
        assert 0.0 <= metrics["recall@10"] <= 1.0
    assert result["configs"]["hybrid"]["recall@10"] > 0


def test_refusal_metrics() -> None:
    rows = [
        {"answerable": True, "refused": False},
        {"answerable": True, "refused": True},
        {"answerable": False, "refused": True},
        {"answerable": False, "refused": False},
    ]
    m = ev.refusal_metrics(rows)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert ev.refusal_metrics([])["precision"] is None


def test_gate_sweep_uses_max_rerank_score() -> None:
    rows = [
        {"answerable": True, "max_rerank": 3.0},
        {"answerable": False, "max_rerank": -6.0},
    ]
    sweep = ev.gate_sweep(rows, thresholds=[-8.0, 0.0])
    assert sweep[0]["threshold"] == -8.0 and sweep[0]["recall"] == 0.0
    assert sweep[1]["precision"] == 1.0 and sweep[1]["recall"] == 1.0


class ScriptedClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], model: str) -> LLMResult:
        self.calls += 1
        ids = [
            ln.split("chunk_id: ")[1].split("]")[0]
            for ln in messages[1]["content"].splitlines()
            if ln.startswith("[chunk_id:")
        ]
        if "Australia" in messages[1]["content"]:
            return LLMResult("NOT_IN_SOURCES", 10, 2, 0.001, model)
        if "roux" in messages[1]["content"].lower() and "flour" in messages[1]["content"].lower():
            return LLMResult(f"Roux [cite: {ids[0]}]", 10, 5, 0.001, model)
        return LLMResult("bad [cite: nope-0000-deadbeef]", 10, 5, 0.001, model)


def fake_scorer(samples: list[dict[str, Any]]) -> dict[str, list[float]]:
    n = len(samples)
    return {
        "faithfulness": [1.0] * n,
        "answer_relevancy": [0.5] * n,
        "context_precision": [0.25] * n,
    }


def test_live_run_aggregates_everything(world: Any, tmp_path: Path) -> None:
    chunks, backend, records = world
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=5.0)
    out = ev.run_live(
        records,
        backend,
        NoopReranker(),
        k=3,
        client=ScriptedClient(),
        model="m",
        scorer=fake_scorer,
        ledger=ledger,
        gen_configs=["hybrid_rerank"],
    )
    cfg = out["configs"]["hybrid_rerank"]
    assert cfg["n"] == len(records)
    assert 0.0 <= cfg["citation_validity_rate"] <= 1.0
    assert 0 < cfg["cost_per_query_usd"] <= 0.001
    assert cfg["latency_ms"]["p95"] >= cfg["latency_ms"]["p50"] >= 0
    assert cfg["ragas"]["faithfulness"] == 1.0
    assert out["refusal"]["recall"] == 1.0  # the Australia question is refused
    assert out["spend_usd"] > 0 and "modern_source_rate" in cfg


def test_live_run_respects_budget_before_calling(world: Any, tmp_path: Path) -> None:
    from src.budget import BudgetExceeded

    _, backend, records = world
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=0.0001)
    client = ScriptedClient()
    with pytest.raises(BudgetExceeded):
        ev.run_live(
            records,
            backend,
            NoopReranker(),
            k=3,
            client=client,
            model="m",
            scorer=fake_scorer,
            ledger=ledger,
            gen_configs=["hybrid_rerank"],
            projected_usd=1.0,
        )
    assert client.calls == 0


def test_markdown_table_and_main_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "res.json"
    code = ev.main(
        [
            "--golden",
            str(FIXTURES / "golden_fixture.jsonl"),
            "--chunks-dir",
            str(tmp_path / "p"),
            "--source",
            str(FIXTURES / "corpus"),
            "--embedder",
            "hash",
            "--reranker",
            "none",
            "--tokenizer",
            "words",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text())
    assert data["smoke"] is True and "ablation" in data and data["generation"] is None
    text = capsys.readouterr().out
    assert "| config |" in text and "hybrid_rerank" in text and "SMOKE" in text


def test_ragas_spend_callback_records(tmp_path: Path) -> None:
    from types import SimpleNamespace

    ledger = SpendLedger(tmp_path / "s.json", cap_usd=5.0)
    cb = ev.SpendCallback(ledger, "judge-model")
    resp = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
    )
    cb.on_llm_end(resp)
    assert ledger.total_usd > 0 and ledger.entries[0]["label"] == "ragas_judge"
