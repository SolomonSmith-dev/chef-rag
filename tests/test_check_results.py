"""Tests for the eval done check."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_results.py"
spec = importlib.util.spec_from_file_location("check_results", SCRIPT)
assert spec and spec.loader
cr = importlib.util.module_from_spec(spec)
sys.modules["check_results"] = cr
spec.loader.exec_module(cr)


def _write(d: Path, name: str, **fields: object) -> None:
    (d / name).write_text(json.dumps(fields))


def test_fails_when_only_smoke(tmp_path: Path) -> None:
    _write(tmp_path, "2026-01-01-smoke.json", smoke=True, ablation={"n_questions": 60})
    assert cr.check(tmp_path, live=False)
    assert cr.main(["--dir", str(tmp_path)]) == 1


def test_fails_when_too_few_labeled_questions(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", smoke=False, ablation={"n_questions": 3})
    assert any("labeled" in p for p in cr.check(tmp_path, live=False))


def test_passes_offline_and_live(tmp_path: Path) -> None:
    gen = {
        "spend_total_all_runs_usd": 1.0,
        "cap_usd": 5.0,
        "configs": {"hybrid_rerank": {"ragas_n_scored": 10}},
    }
    _write(tmp_path, "a.json", smoke=False, ablation={"n_questions": 50}, generation=gen)
    assert cr.check(tmp_path, live=True) == []
    _write(tmp_path, "b.json", smoke=False, ablation={"n_questions": 50}, generation=None)
    assert cr.check(tmp_path, live=True)  # latest (b) lacks generation
