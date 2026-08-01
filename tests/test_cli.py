"""Tests for the CLI."""

from __future__ import annotations

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
