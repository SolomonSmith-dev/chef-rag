"""Tests for deploy and reporting scripts: prepare_space, check_demo, render_results."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from fastapi.testclient import TestClient

from src.demo import create_app
from src.generate import Answer
from src.retrieval import Hit
from src.trace import JsonlTracer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REPO = SCRIPTS.parent


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prepare_space = _load("prepare_space")
check_demo = _load("check_demo")
render_results = _load("render_results")


def test_prepare_space_copies_needed_files_and_no_secrets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "demo.py").write_text("x")
    (repo / "src" / ".env").write_text("SECRET=1")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "fetch_corpus.py").write_text("y")
    (repo / "deploy" / "hf-space").mkdir(parents=True)
    (repo / "deploy" / "hf-space" / "Dockerfile").write_text("FROM x")
    (repo / "deploy" / "hf-space" / "README.md").write_text("---\nsdk: docker\n---")
    (repo / "pyproject.toml").write_text("[project]")
    (repo / "data" / "raw" / "original").mkdir(parents=True)
    (repo / "data" / "raw" / "original" / "note.md").write_text("mine")
    (repo / "data" / "raw" / "original" / "README.md").write_text("skip me")
    out = tmp_path / "out"
    copied = prepare_space.prepare(out, repo)
    assert (out / "Dockerfile").is_file() and (out / "src" / "demo.py").is_file()
    assert not list(out.rglob(".env"))
    assert "data/raw/original/note.md" in copied
    assert not (out / "data" / "raw" / "original" / "README.md").exists()


def test_real_dockerfile_and_space_readme_are_consistent() -> None:
    dockerfile = (REPO / "deploy" / "hf-space" / "Dockerfile").read_text()
    readme = (REPO / "deploy" / "hf-space" / "README.md").read_text()
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    assert "EXPOSE 7860" in dockerfile and "--factory" in dockerfile
    assert "fetch_corpus.py --verify" in dockerfile
    assert "OPENROUTER_API_KEY=" not in dockerfile  # secret must come from the Space


class _Engine:
    def retrieve(self, q: str) -> list[Hit]:
        return [Hit("a-0", "t", "p", "T", "fda", 0.03, 1, 1, 2.0)]

    def generate(self, q: str, hits: list[Hit]) -> Answer:
        return Answer("ok [cite: a-0]", ["a-0"], False, None, hits, "m", 10, 5, 0.0)


def test_check_demo_passes_against_good_app_and_fails_on_bad(tmp_path: Path) -> None:
    app = create_app(_Engine(), JsonlTracer(tmp_path / "t", redact_input=True))
    assert check_demo.run_checks(TestClient(app), rate_limit=True) == []

    class Broken(_Engine):
        def retrieve(self, q: str) -> list[Hit]:
            return []

    bad = create_app(Broken(), JsonlTracer(tmp_path / "t2", redact_input=True))
    assert any("scores" in p for p in check_demo.run_checks(TestClient(bad)))


RESULT = {
    "smoke": False,
    "ablation": {
        "k": 5,
        "n_questions": 50,
        "unresolved": ["x"],
        "configs": {
            "dense": {"recall@10": 0.5, "recall@k": 0.4, "mrr": 0.3},
            "hybrid_rerank": {"recall@10": 0.8, "recall@k": 0.7, "mrr": 0.6},
        },
    },
    "generation": {
        "configs": {
            "hybrid_rerank": {
                "ragas": {"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7},
                "refusal": {"precision": 1.0, "recall": 0.75},
                "citation_validity_rate": 1.0,
                "cost_per_query_usd": 0.004,
                "latency_ms": {"p50": 1200.0, "p95": 2500.0},
            }
        }
    },
}


def test_render_results_pending_and_real(tmp_path: Path) -> None:
    readme = (REPO / "README.md").read_text()
    roadmap = (REPO / "ROADMAP.md").read_text()
    pending_r, pending_m = render_results.build(readme, roadmap, tmp_path)
    assert "Not yet run" in pending_r and "not%20run" in pending_r
    (tmp_path / "2026-02-01.json").write_text(json.dumps(RESULT))
    (tmp_path / "2026-03-01-smoke.json").write_text(json.dumps({**RESULT, "smoke": True}))
    new_r, new_m = render_results.build(readme, roadmap, tmp_path)
    assert "| hybrid_rerank | 0.800 | 0.700 | 0.600 | 0.900 | 0.800 | 0.700 |" in new_r
    assert "Refusal precision 1.000, recall 0.750" in new_r
    assert "recall%405" in new_r or "recall@5" in new_r
    assert "| v1 baseline | 0.900 | 0.800 | 0.700 |" in new_m


def test_committed_readme_matches_committed_results() -> None:
    assert render_results.main(["--check"]) == 0


def test_readme_section_order() -> None:
    text = (REPO / "README.md").read_text()
    order = [
        "## Why this domain",
        "## Demo",
        "## Results",
        "## Architecture",
        "## What I'd do next",
        "## Setup",
    ]
    positions = [text.index(h) for h in order]
    assert positions == sorted(positions)
    assert "in%20development" not in text
