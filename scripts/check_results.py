"""Done check for eval runs. Exits non-zero unless a real (non-smoke) result exists.

uv run python scripts/check_results.py            # offline ablation present
uv run python scripts/check_results.py --live     # generation + Ragas present, under cap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def latest_real_result(results_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted(results_dir.glob("*.json"), reverse=True):
        data = json.loads(path.read_text())
        if not data.get("smoke", True):
            return path, data
    return None


def check(results_dir: Path, live: bool, min_questions: int = 40) -> list[str]:
    found = latest_real_result(results_dir)
    if found is None:
        return [f"no non-smoke result in {results_dir}"]
    path, data = found
    problems: list[str] = []
    ablation = data.get("ablation", {})
    if ablation.get("n_questions", 0) < min_questions:
        problems.append(
            f"{path.name}: only {ablation.get('n_questions', 0)} labeled questions "
            f"(need {min_questions}); is the corpus downloaded?"
        )
    if live:
        gen = data.get("generation")
        if not gen:
            problems.append(f"{path.name}: no live generation section")
        elif gen["spend_total_all_runs_usd"] > gen["cap_usd"]:
            problems.append(f"{path.name}: spend over cap")
        else:
            for name, cfg in gen["configs"].items():
                if cfg["ragas_n_scored"] == 0:
                    problems.append(f"{path.name}: {name} has no Ragas scores")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "evals" / "results")
    args = ap.parse_args(argv)
    problems = check(args.dir, args.live)
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if not problems:
        print("OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
