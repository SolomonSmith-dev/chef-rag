"""Eval harness: retrieval ablation (offline) and generation quality (live, capped).

One command reproduces the headline table:

    uv run python evals/run_evals.py            # offline ablation, no LLM calls
    uv run python evals/run_evals.py --live     # + generation, Ragas, refusal, cost

Retrieval labels are resolved at run time: a chunk is relevant to a question when it
comes from one of the question's ``sources`` and matches one of its ``evidence``
regexes. Questions that resolve to zero chunks are listed under ``unresolved`` and
excluded from retrieval metrics, never silently scored as zero.

Recall@N is |relevant in top N| / min(|relevant|, N), so a question with many relevant
chunks is not capped below 1.0 by N.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from src.budget import SpendLedger  # noqa: E402
from src.generate import (  # noqa: E402
    OPENROUTER_URL,
    CitationError,
    LLMClient,
    answer_question,
    estimate_cost,
)
from src.ingest import Chunk  # noqa: E402
from src.rerank import Reranker  # noqa: E402
from src.retrieval import Hit, LocalBackend, RetrievalBackend  # noqa: E402
from src.trace import Tracer  # noqa: E402

CONFIGS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
MODERN_SOURCES = {"fda", "usda"}
RAGAS_KEYS = ["faithfulness", "answer_relevancy", "context_precision"]
Scorer = Callable[[list[dict[str, Any]]], dict[str, list[float]]]


# --- data and labels -------------------------------------------------------------


def sample_records(records: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    """First ``n`` records taken round-robin across categories, so a small live run
    still covers every category. ``None`` returns all records."""
    if n is None or n >= len(records):
        return records
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        buckets.setdefault(rec["category"], []).append(rec)
    picked: list[dict[str, Any]] = []
    while len(picked) < n:
        for bucket in buckets.values():
            if bucket and len(picked) < n:
                picked.append(bucket.pop(0))
    return picked


def load_golden(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve_relevant(record: dict[str, Any], chunks: list[Chunk]) -> set[str]:
    """Chunk ids that count as relevant for ``record`` (empty for unanswerable questions)."""
    if not record.get("answerable"):
        return set()
    patterns = [re.compile(p, re.I | re.S) for p in record.get("evidence", [])]
    sources = record.get("sources", [])
    relevant: set[str] = set()
    for chunk in chunks:
        in_source = any(
            chunk.source_type == "original" if s == "original" else chunk.source_path.startswith(s)
            for s in sources
        )
        if in_source and any(p.search(chunk.text) for p in patterns):
            relevant.add(chunk.chunk_id)
    return relevant


# --- metrics ---------------------------------------------------------------------


def recall_at(relevant: set[str], ranked: list[str], n: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:n])) / min(len(relevant), n)


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    for i, cid in enumerate(ranked, 1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile."""
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * pct / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def refusal_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Precision: refusals that were right. Recall: unanswerable questions that were refused."""
    refused = [r for r in rows if r["refused"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    correct = [r for r in refused if not r["answerable"]]
    return {
        "precision": len(correct) / len(refused) if refused else None,
        "recall": len(correct) / len(unanswerable) if unanswerable else None,
        "n_refused": len(refused),
        "n_unanswerable": len(unanswerable),
    }


def gate_sweep(
    rows: list[dict[str, Any]], thresholds: list[float] | None = None
) -> list[dict[str, float | None]]:
    """Refusal precision/recall if the retrieval gate were set at each threshold."""
    thresholds = thresholds if thresholds is not None else [x / 2 for x in range(-16, 9)]
    out: list[dict[str, float | None]] = []
    for t in thresholds:
        sim = [{"answerable": r["answerable"], "refused": r["max_rerank"] < t} for r in rows]
        m = refusal_metrics(sim)
        out.append({"threshold": t, "precision": m["precision"], "recall": m["recall"]})  # type: ignore[dict-item]
    return out


# --- offline retrieval ablation ---------------------------------------------------


def _rankings(
    config: str, query: str, backend: RetrievalBackend, reranker: Reranker
) -> tuple[list[Hit], list[Hit]]:
    """Return (candidate list before rerank, final ranked list), both up to 10 hits."""
    if config == "bm25":
        hits = backend.search(query, mode="bm25", limit=10)
        return hits, hits
    if config == "dense":
        hits = backend.search(query, mode="dense", limit=10)
        return hits, hits
    hits = backend.search(query, mode="hybrid", limit=10)
    if config == "hybrid":
        return hits, hits
    return hits, reranker.rerank(query, hits, top_k=10)


def run_ablation(
    records: list[dict[str, Any]],
    chunks: list[Chunk],
    backend: RetrievalBackend,
    reranker: Reranker,
    k: int = 5,
) -> dict[str, Any]:
    scored: list[tuple[dict[str, Any], set[str]]] = []
    unresolved: list[str] = []
    for rec in records:
        if not rec.get("answerable"):
            continue
        rel = resolve_relevant(rec, chunks)
        if rel:
            scored.append((rec, rel))
        else:
            unresolved.append(rec["id"])
    configs: dict[str, dict[str, float | None]] = {}
    for config in CONFIGS:
        r10: list[float] = []
        rk: list[float] = []
        rr: list[float] = []
        for rec, rel in scored:
            before, final = _rankings(config, rec["question"], backend, reranker)
            r10.append(recall_at(rel, [h.chunk_id for h in before], 10))
            final_ids = [h.chunk_id for h in final]
            rk.append(recall_at(rel, final_ids, k))
            rr.append(reciprocal_rank(rel, final_ids))
        configs[config] = {
            "recall@10": _mean(r10) if r10 else None,
            "recall@k": _mean(rk) if rk else None,
            "mrr": _mean(rr) if rr else None,
        }
    gate_rows = []
    for rec in records:
        _, final = _rankings("hybrid_rerank", rec["question"], backend, reranker)
        scores = [h.rerank_score for h in final[:k] if h.rerank_score is not None]
        if scores:
            gate_rows.append({"answerable": bool(rec.get("answerable")), "max_rerank": max(scores)})
    return {
        "k": k,
        "n_questions": len(scored),
        "unresolved": unresolved,
        "configs": configs,
        "gate_sweep": gate_sweep(gate_rows) if gate_rows else None,
    }


# --- live generation, Ragas, refusal, cost -----------------------------------------


class SpendCallback(BaseCallbackHandler):
    """Records Ragas judge token usage in the spend ledger (cost is estimated)."""

    def __init__(self, ledger: SpendLedger, model: str) -> None:
        self.ledger, self.model = ledger, model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {})
        p, c = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        self.ledger.record("ragas_judge", self.model, p, c, estimate_cost(self.model, p, c))


def make_ragas_scorer(
    judge_model: str, api_key: str, ledger: SpendLedger, embedding_model: str
) -> Scorer:
    """Ragas faithfulness, answer relevancy and context precision via OpenRouter."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model,
            base_url=OPENROUTER_URL,
            api_key=api_key,  # type: ignore[arg-type]
            temperature=0,
            callbacks=[SpendCallback(ledger, judge_model)],
        )
    )
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model))

    def score(samples: list[dict[str, Any]]) -> dict[str, list[float]]:
        ledger.check()
        dataset = EvaluationDataset(samples=[SingleTurnSample(**s) for s in samples])
        result = evaluate(
            dataset,
            metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()],
            llm=llm,
            embeddings=emb,
            show_progress=False,
        )
        out: dict[str, list[float]] = {key: [] for key in RAGAS_KEYS}
        for row in result.scores:  # type: ignore[attr-defined]
            for key in RAGAS_KEYS:
                match = [v for name, v in row.items() if key in name]
                if match and match[0] is not None and not math.isnan(match[0]):
                    out[key].append(float(match[0]))
        return out

    return score


def _final_hits(
    config: str, query: str, backend: RetrievalBackend, reranker: Reranker, k: int
) -> list[Hit]:
    _, final = _rankings(config, query, backend, reranker)
    return final[:k]


def run_live(
    records: list[dict[str, Any]],
    backend: RetrievalBackend,
    reranker: Reranker,
    k: int,
    client: LLMClient,
    model: str,
    scorer: Scorer,
    ledger: SpendLedger,
    gen_configs: list[str],
    projected_usd: float = 0.0,
    tracer: Tracer | None = None,
    min_rerank_score: float | None = 0.0,
) -> dict[str, Any]:
    ledger.ensure_room(projected_usd)
    judge_start = len(ledger.entries)
    out_configs: dict[str, Any] = {}
    total_cost = 0.0
    for config in gen_configs:
        rows: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        for rec in records:
            t0 = time.perf_counter()
            hits = _final_hits(config, rec["question"], backend, reranker, k)
            row: dict[str, Any] = {
                "id": rec["id"],
                "category": rec["category"],
                "answerable": bool(rec.get("answerable")),
                "refused": False,
                "citation_ok": None,
                "modern_source": None,
                "cost": 0.0,
            }
            try:
                ans = answer_question(
                    rec["question"], hits, client, model, min_rerank_score, tracer
                )
            except CitationError as exc:
                row["citation_ok"] = False
                row["error"] = type(exc).__name__
                # cost of the failed call is in the client ledger; count 0 here
            else:
                row.update(refused=ans.refused, cost=ans.cost)
                if not ans.refused:
                    row["citation_ok"] = True
                    kinds = {h.source_type for h in hits if h.chunk_id in ans.citations}
                    row["modern_source"] = bool(kinds & MODERN_SOURCES)
                    if row["answerable"]:
                        samples.append(
                            {
                                "user_input": rec["question"],
                                "response": ans.text,
                                "retrieved_contexts": [h.text for h in hits],
                                "reference": rec["reference_answer"],
                            }
                        )
                        sample_rows.append(row)
            row["latency_ms"] = (time.perf_counter() - t0) * 1000
            rows.append(row)
        scores = scorer(samples) if samples else {k_: [] for k_ in RAGAS_KEYS}
        answered = [r for r in rows if r["citation_ok"] is not None]
        outdated = [
            r for r in rows if r["category"] == "outdated" and r["modern_source"] is not None
        ]
        cost = sum(r["cost"] for r in rows)
        total_cost += cost
        out_configs[config] = {
            "n": len(rows),
            "citation_validity_rate": (
                sum(1 for r in answered if r["citation_ok"]) / len(answered) if answered else None
            ),
            "modern_source_rate": (
                sum(1 for r in outdated if r["modern_source"]) / len(outdated) if outdated else None
            ),
            "refusal": refusal_metrics(rows),
            "latency_ms": {
                "p50": percentile([r["latency_ms"] for r in rows], 50),
                "p95": percentile([r["latency_ms"] for r in rows], 95),
            },
            "cost_per_query_usd": cost / len(rows) if rows else 0.0,
            "ragas": {key: _mean(vals) if vals else None for key, vals in scores.items()},
            "ragas_n_scored": len(samples),
            "per_question": [{k_: v for k_, v in r.items() if k_ != "latency_ms"} for r in rows],
        }
    judge_cost = sum(e["usd"] for e in ledger.entries[judge_start:] if e["label"] == "ragas_judge")
    shipped = out_configs.get("hybrid_rerank") or next(iter(out_configs.values()))
    return {
        "model": model,
        "configs": out_configs,
        "refusal": shipped["refusal"],
        "generation_cost_usd": total_cost,
        "judge_cost_usd_estimated": judge_cost,
        "spend_usd": total_cost + judge_cost,
        "spend_total_all_runs_usd": ledger.total_usd,
        "cap_usd": ledger.cap_usd,
    }


def estimate_live_cost(
    n_queries: int, n_configs: int, k: int, model: str, judge_model: str
) -> float:
    """Pre-flight estimate in USD. Deliberately pessimistic so the cap is never surprised."""
    gen_in, gen_out = k * 650 + 450, 350
    gen = n_queries * n_configs * estimate_cost(model, gen_in, gen_out)
    judge = n_queries * n_configs * 0.8 * estimate_cost(judge_model, 15_000, 1_800)
    return gen + judge


# --- reporting -------------------------------------------------------------------


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def markdown_table(ablation: dict[str, Any], generation: dict[str, Any] | None, smoke: bool) -> str:
    gen = (generation or {}).get("configs", {})
    lines = []
    if smoke:
        lines.append("SMOKE RUN: not a quality result (stand-in embedder or no reranker).")
    lines += [
        f"| config | recall@10 (pre-rerank) | recall@{ablation['k']} | MRR | faithfulness "
        "| answer relevancy | context precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in ablation["configs"].items():
        rag = gen.get(name, {}).get("ragas", {})
        lines.append(
            f"| {name} | {_fmt(m['recall@10'])} | {_fmt(m['recall@k'])} | {_fmt(m['mrr'])} | "
            f"{_fmt(rag.get('faithfulness'))} | {_fmt(rag.get('answer_relevancy'))} | "
            f"{_fmt(rag.get('context_precision'))} |"
        )
    return "\n".join(lines)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    ap.add_argument(
        "--source", type=Path, help="ingest this dir on the fly instead of using a built index"
    )
    ap.add_argument("--chunks-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    ap.add_argument("--reranker", choices=["bge", "none"], default="bge")
    ap.add_argument("--tokenizer", choices=["cl100k", "words"], default="cl100k")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--live", action="store_true", help="run generation + Ragas (spends money)")
    ap.add_argument("--gen-configs", default="dense,hybrid_rerank")
    ap.add_argument(
        "--model", default=os.environ.get("GENERATION_MODEL", "anthropic/claude-haiku-4.5")
    )
    ap.add_argument("--judge-model", default="anthropic/claude-haiku-4.5")
    ap.add_argument(
        "--limit", type=int, help="live run only: use N questions, spread across categories"
    )
    ap.add_argument("--cap-usd", type=float, default=5.0)
    ap.add_argument("--ledger", type=Path, default=Path(".traces/spend.json"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from src.config import load_retrieval_settings
    from src.ingest import ingest_directory, load_chunks
    from src.pipeline import build_embedder, build_reranker

    cfg = load_retrieval_settings()
    embedder = build_embedder(args.embedder, cfg)
    if args.source:
        from src.ingest import TiktokenTokenizer, WordTokenizer

        tok = WordTokenizer() if args.tokenizer == "words" else TiktokenTokenizer()
        ingest_directory(args.source, args.chunks_dir, tokenizer=tok)
        chunks = load_chunks(args.chunks_dir / "chunks.jsonl")
        backend: LocalBackend = LocalBackend.build(chunks, embedder)
    else:
        backend = LocalBackend.load(Path(cfg.index_dir), embedder)
        chunks = backend.chunks
    reranker = build_reranker(args.reranker, cfg)
    records = load_golden(args.golden)

    ablation = run_ablation(records, chunks, backend, reranker, args.k)
    reasons = []
    if args.embedder == "hash":
        reasons.append("hash embedder is lexical, not semantic")
    if args.reranker == "none":
        reasons.append("no cross-encoder reranker")
    smoke = bool(reasons)

    generation: dict[str, Any] | None = None
    ledger = SpendLedger(args.ledger, args.cap_usd)
    if args.live:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
            return 2
        from src.generate import OpenRouterClient
        from src.trace import make_tracer

        configs = args.gen_configs.split(",")
        live_records = sample_records(records, args.limit)
        projected = estimate_live_cost(
            len(live_records), len(configs), args.k, args.model, args.judge_model
        )
        print(f"projected live cost ${projected:.2f}; remaining ${ledger.remaining_usd:.2f}")
        tracer = make_tracer(os.environ)
        generation = run_live(
            live_records,
            backend,
            reranker,
            args.k,
            OpenRouterClient(key, ledger, max_tokens=600),
            args.model,
            make_ragas_scorer(args.judge_model, key, ledger, cfg.local_embedding_model),
            ledger,
            configs,
            projected,
            tracer,
            cfg.min_rerank_score,
        )
        tracer.flush()

    result = {
        "date": date.today().isoformat(),
        "git_sha": _git_sha(),
        "smoke": smoke,
        "smoke_reasons": reasons,
        "config": {
            "embedder": args.embedder,
            "embedding_model": cfg.local_embedding_model,
            "reranker": args.reranker,
            "rerank_model": cfg.rerank_model,
            "k": args.k,
            "rrf_k": 60,
            "candidates_per_retriever": 20,
            "fuse_top_n": 10,
        },
        "corpus": {"n_chunks": len(chunks), "n_golden": len(records)},
        "ablation": ablation,
        "generation": generation,
    }
    out = (
        args.out
        or Path("evals/results") / f"{date.today().isoformat()}{'-smoke' if smoke else ''}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(markdown_table(ablation, generation, smoke))
    print(f"unresolved questions (no labeled chunks): {len(ablation['unresolved'])}")
    if generation:
        print(f"live spend ${generation['spend_usd']:.4f} (judge cost is an estimate)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
