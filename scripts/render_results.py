"""Render the latest real eval result into README.md and ROADMAP.md.

    uv run python scripts/render_results.py          # rewrite the marked blocks
    uv run python scripts/render_results.py --check  # exit 1 if they are stale

Only non-smoke results count. With none, the blocks say "not yet run".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_results import latest_real_result  # noqa: E402

PENDING_TABLE = (
    "_Not yet run. The first real run needs the downloaded corpus, model weights and an "
    "OpenRouter key: see [docs/eval-runbook.md](docs/eval-runbook.md)._"
)


def _f(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_table(data: dict[str, Any] | None, name: str | None = None) -> str:
    if data is None:
        return PENDING_TABLE
    abl = data["ablation"]
    gen = (data.get("generation") or {}).get("configs", {})
    k = abl["k"]
    lines = [
        f"| config | recall@10 (pre-rerank) | recall@{k} | MRR | faithfulness | answer relevancy "
        "| context precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg, m in abl["configs"].items():
        rag = gen.get(cfg, {}).get("ragas", {})
        lines.append(
            f"| {cfg} | {_f(m['recall@10'])} | {_f(m['recall@k'])} | {_f(m['mrr'])} | "
            f"{_f(rag.get('faithfulness'))} | {_f(rag.get('answer_relevancy'))} | "
            f"{_f(rag.get('context_precision'))} |"
        )
    shipped = gen.get("hybrid_rerank")
    extra = (
        f"{abl['n_questions']} labeled questions, {len(abl['unresolved'])} unresolved. "
        f"Source: `evals/results/{name}`."
    )
    if shipped:
        r = shipped["refusal"]
        extra += (
            f" Refusal precision {_f(r['precision'])}, recall {_f(r['recall'])}; "
            f"citation validity {_f(shipped['citation_validity_rate'])}; "
            f"cost per query ${shipped['cost_per_query_usd']:.4f}; "
            f"latency p50 {shipped['latency_ms']['p50']:.0f} ms, "
            f"p95 {shipped['latency_ms']['p95']:.0f} ms."
        )
    return "\n".join(lines) + "\n\n" + extra


def render_badge(data: dict[str, Any] | None) -> str:
    if data is None:
        label, color = "not run", "lightgrey"
    else:
        hr = data["ablation"]["configs"].get("hybrid_rerank", {})
        label, color = (
            f"hybrid+rerank recall@{data['ablation']['k']} {_f(hr.get('recall@k'))}",
            "blue",
        )
    return f"![Eval](https://img.shields.io/badge/eval-{quote(label, safe='')}-{color})"


def _replace(text: str, tag: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {tag}:start -->\n).*?(\n<!-- {tag}:end -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"marker {tag} not found")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def render_roadmap_table(data: dict[str, Any] | None) -> str:
    head = "| Version | Faithfulness | Answer Relevancy | Context Precision |\n|---|---|---|---|\n"
    if data is None:
        return head + "| v1 baseline | _not run_ | _not run_ | _not run_ |"
    gen = (
        (data.get("generation") or {}).get("configs", {}).get("hybrid_rerank", {}).get("ragas", {})
    )
    return head + (
        f"| v1 baseline | {_f(gen.get('faithfulness'))} | {_f(gen.get('answer_relevancy'))} "
        f"| {_f(gen.get('context_precision'))} |"
    )


def build(readme: str, roadmap: str, results_dir: Path) -> tuple[str, str]:
    found = latest_real_result(results_dir)
    path, data = (found[0].name, found[1]) if found else (None, None)
    readme = _replace(readme, "ablation", render_table(data, path))
    readme = _replace(readme, "eval-badge", render_badge(data))
    roadmap = _replace(roadmap, "baseline", render_roadmap_table(data))
    return readme, roadmap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "evals" / "results")
    args = ap.parse_args(argv)
    readme_path, roadmap_path = REPO_ROOT / "README.md", REPO_ROOT / "ROADMAP.md"
    old_r, old_m = readme_path.read_text(), roadmap_path.read_text()
    new_r, new_m = build(old_r, old_m, args.results)
    if args.check:
        if (new_r, new_m) != (old_r, old_m):
            print("FAIL README/ROADMAP are stale; run scripts/render_results.py", file=sys.stderr)
            return 1
        print("OK")
        return 0
    readme_path.write_text(new_r)
    roadmap_path.write_text(new_m)
    print("updated README.md and ROADMAP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
