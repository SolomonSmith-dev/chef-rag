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


def test_cli_query_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["query", "how do I hold a chef knife"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not implemented" in captured.err


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
