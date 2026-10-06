"""Tracing. Every request emits a trace: Langfuse when keys exist, else local JSONL.

``make_tracer`` is the only place that chooses a backend, so no code path can run
untraced. ``redact_input`` drops inputs and outputs (used by the public demo, which
must never log question text).
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Kind = Literal["span", "generation"]
_current: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "chef_rag_span", default=None
)


class SpanHandle(ABC):
    @abstractmethod
    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage: Mapping[str, int] | None = None,
        cost: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...


class Tracer(ABC):
    @abstractmethod
    def span(
        self,
        name: str,
        *,
        kind: Kind = "span",
        input: Any = None,  # noqa: A002
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Context manager yielding a ``SpanHandle``."""

    def flush(self) -> None:  # noqa: B027 - optional hook
        pass


class _JsonlSpan(SpanHandle):
    def __init__(self, redact: bool) -> None:
        self.fields: dict[str, Any] = {}
        self._redact = redact

    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage: Mapping[str, int] | None = None,
        cost: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if output is not None and not self._redact:
            self.fields["output"] = output
        if model is not None:
            self.fields["model"] = model
        if usage is not None:
            self.fields["usage"] = dict(usage)
        if cost is not None:
            self.fields["cost"] = cost
        if metadata:
            self.fields.setdefault("metadata", {}).update(metadata)


class JsonlTracer(Tracer):
    """Appends one JSON line per finished span to ``<dir>/<date>.jsonl``."""

    def __init__(self, directory: Path, redact_input: bool = False) -> None:
        self.directory = directory
        self.redact_input = redact_input

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: Kind = "span",
        input: Any = None,  # noqa: A002
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        parent = _current.get()
        trace_id = parent[0] if parent else uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        handle = _JsonlSpan(self.redact_input)
        token = _current.set((trace_id, span_id))
        start = time.perf_counter()
        started_at = datetime.now(UTC)
        error: str | None = None
        try:
            yield handle
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _current.reset(token)
            record: dict[str, Any] = {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_id": parent[1] if parent else None,
                "name": name,
                "kind": kind,
                "start": started_at.isoformat(),
                "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                "error": error,
                **handle.fields,
            }
            if input is not None and not self.redact_input:
                record["input"] = input
            if metadata:
                record["metadata"] = {**record.get("metadata", {}), **metadata}
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{started_at:%Y-%m-%d}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class _LangfuseSpan(SpanHandle):
    def __init__(self, obs: Any, redact: bool) -> None:
        self._obs = obs
        self._redact = redact

    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage: Mapping[str, int] | None = None,
        cost: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None and not self._redact:
            kwargs["output"] = output
        if model is not None:
            kwargs["model"] = model
        if usage is not None:
            kwargs["usage_details"] = {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }
        if cost is not None:
            kwargs["cost_details"] = {"total": cost}
        if metadata:
            kwargs["metadata"] = dict(metadata)
        self._obs.update(**kwargs)


class LangfuseTracer(Tracer):
    """Langfuse v4 client: ``start_as_current_observation`` nests spans automatically."""

    def __init__(self, client: Any, redact_input: bool = False) -> None:
        self._client = client
        self._redact = redact_input

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: Kind = "span",
        input: Any = None,  # noqa: A002
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        with self._client.start_as_current_observation(
            name=name,
            as_type=kind,
            input=None if self._redact else input,
            metadata=dict(metadata or {}),
        ) as obs:
            yield _LangfuseSpan(obs, self._redact)

    def flush(self) -> None:
        self._client.flush()


def make_tracer(
    env: Mapping[str, str],
    trace_dir: Path | None = None,
    redact_input: bool = False,
    langfuse_factory: Callable[[], Any] | None = None,
) -> Tracer:
    """Langfuse when both keys are present, otherwise local JSONL in ``trace_dir``."""
    if env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"):
        if langfuse_factory is None:

            def langfuse_factory() -> Any:
                from langfuse import Langfuse

                return Langfuse(
                    public_key=env["LANGFUSE_PUBLIC_KEY"],
                    secret_key=env["LANGFUSE_SECRET_KEY"],
                    host=env.get("LANGFUSE_HOST") or None,
                )

        return LangfuseTracer(langfuse_factory(), redact_input=redact_input)
    return JsonlTracer(
        trace_dir or Path(env.get("TRACE_DIR", ".traces")), redact_input=redact_input
    )
