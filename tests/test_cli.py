"""Tests for the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "chef-rag" in captured.out


def test_cli_query_without_index_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "missing"))
    exit_code = main(["query", "knife", "--embedder", "hash", "--reranker", "none"])
    assert exit_code == 2
    assert "ingest --index" in capsys.readouterr().err


def test_cli_ingest_index_then_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "corpus"
    out = tmp_path / "processed"
    args = ["--source", str(fixtures), "--out", str(out), "--tokenizer", "words"]
    assert main(["ingest", *args, "--index", "--embedder", "hash"]) == 0
    monkeypatch.setenv("INDEX_DIR", str(out / "index"))
    code = main(
        [
            "query",
            "poultry internal temperature",
            "--embedder",
            "hash",
            "--reranker",
            "none",
            "--show-scores",
            "--k",
            "2",
        ]
    )
    out_text = capsys.readouterr().out
    assert code == 0
    assert "rrf=" in out_text and "fixture-fda-foodcode" in out_text


def test_cli_ingest_fixture_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "corpus"
    code = main(
        ["ingest", "--source", str(fixtures), "--out", str(tmp_path), "--tokenizer", "words"]
    )
    assert code == 0
    assert (tmp_path / "chunks.jsonl").is_file()
    assert "chunks" in capsys.readouterr().out


def test_cli_ingest_missing_source(tmp_path: Path) -> None:
    assert main(["ingest", "--source", str(tmp_path / "nope")]) == 2


def test_cli_query_emits_local_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "corpus"
    out = tmp_path / "processed"
    base = ["--source", str(fixtures), "--out", str(out), "--tokenizer", "words"]
    assert main(["ingest", *base, "--index", "--embedder", "hash"]) == 0
    monkeypatch.setenv("INDEX_DIR", str(out / "index"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert main(["query", "danger zone", "--embedder", "hash", "--reranker", "none"]) == 0
    text = next((tmp_path / "traces").glob("*.jsonl")).read_text()
    assert '"name": "query"' in text and '"name": "retrieve"' in text and '"name": "rerank"' in text


def test_cli_answer_requires_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "corpus"
    out = tmp_path / "processed"
    base = ["--source", str(fixtures), "--out", str(out), "--tokenizer", "words"]
    assert main(["ingest", *base, "--index", "--embedder", "hash"]) == 0
    monkeypatch.setenv("INDEX_DIR", str(out / "index"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code = main(["query", "danger zone", "--answer", "--embedder", "hash", "--reranker", "none"])
    assert code == 2
