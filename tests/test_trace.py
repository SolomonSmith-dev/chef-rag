"""Tests for tracing: Langfuse when keys exist, local JSONL otherwise. No dark calls."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from src.trace import JsonlTracer, LangfuseTracer, make_tracer


def _records(path: Path) -> list[dict[str, Any]]:
    lines = [ln for f in sorted(path.glob("*.jsonl")) for ln in f.read_text().splitlines()]
    return [json.loads(ln) for ln in lines]


def test_jsonl_tracer_records_nested_spans(tmp_path: Path) -> None:
    tracer = JsonlTracer(tmp_path)
    with tracer.span("query", input={"q": "hi"}) as root:
        with tracer.span("generate", kind="generation") as gen:
            gen.update(output="ok", model="m", usage={"prompt_tokens": 3}, cost=0.01)
        root.update(output="done")
    recs = {r["name"]: r for r in _records(tmp_path)}
    assert recs["generate"]["parent_id"] == recs["query"]["span_id"]
    assert recs["generate"]["trace_id"] == recs["query"]["trace_id"]
    assert recs["generate"]["cost"] == 0.01 and recs["generate"]["model"] == "m"
    assert recs["query"]["parent_id"] is None
    assert recs["query"]["duration_ms"] >= 0


def test_jsonl_tracer_records_errors_and_reraises(tmp_path: Path) -> None:
    tracer = JsonlTracer(tmp_path)
    with pytest.raises(RuntimeError), tracer.span("boom"):
        raise RuntimeError("bad")
    assert _records(tmp_path)[0]["error"] == "RuntimeError: bad"


def test_jsonl_tracer_redacts_input(tmp_path: Path) -> None:
    tracer = JsonlTracer(tmp_path, redact_input=True)
    with tracer.span("query", input={"q": "secret question"}) as s:
        s.update(output="secret answer")
    text = (next(tmp_path.glob("*.jsonl"))).read_text()
    assert "secret question" not in text and "secret answer" not in text


class _FakeObservation:
    def __init__(self, log: list[tuple[str, dict[str, Any]]], name: str) -> None:
        self.log, self.name = log, name

    def update(self, **kwargs: Any) -> None:
        self.log.append((self.name, kwargs))


class _FakeLangfuse:
    def __init__(self) -> None:
        self.log: list[tuple[str, dict[str, Any]]] = []
        self.started: list[dict[str, Any]] = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Iterator[_FakeObservation]:
        self.started.append(kwargs)
        yield _FakeObservation(self.log, kwargs["name"])

    def flush(self) -> None:
        self.flushed = True


def test_langfuse_tracer_maps_generation_fields() -> None:
    client = _FakeLangfuse()
    tracer = LangfuseTracer(client)
    with tracer.span("generate", kind="generation", input="q") as s:
        s.update(
            output="a", model="m", usage={"prompt_tokens": 1, "completion_tokens": 2}, cost=0.5
        )
    assert client.started[0]["as_type"] == "generation"
    update = client.log[0][1]
    assert update["output"] == "a" and update["model"] == "m"
    assert update["usage_details"] == {"input": 1, "output": 2}
    assert update["cost_details"] == {"total": 0.5}
    tracer.flush()
    assert client.flushed


def test_make_tracer_selects_by_env(tmp_path: Path) -> None:
    assert isinstance(make_tracer({}, trace_dir=tmp_path), JsonlTracer)
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    tracer = make_tracer(env, trace_dir=tmp_path, langfuse_factory=lambda: _FakeLangfuse())
    assert isinstance(tracer, LangfuseTracer)


def test_langfuse_tracer_redacts() -> None:
    client = _FakeLangfuse()
    tracer = LangfuseTracer(client, redact_input=True)
    with tracer.span("q", input="secret question") as s:
        s.update(output="secret answer")
    assert client.started[0]["input"] is None
    assert "output" not in client.log[0][1]
