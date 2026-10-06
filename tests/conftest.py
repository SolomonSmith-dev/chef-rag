"""Shared test setup: keep traces out of the working tree."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_trace_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACE_DIR", str(tmp_path_factory.mktemp("traces")))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
