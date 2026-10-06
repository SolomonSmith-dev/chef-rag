"""Cross-encoder reranking (bge-reranker-base): top 10 candidates down to top k."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from src.retrieval import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class NoopReranker:
    """Keeps retrieval order. Used for ablations and offline smoke runs."""

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder. ``model`` is injectable for tests."""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-base", model: Any | None = None
    ) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self._model = model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.text) for h in hits])
        scored = [replace(h, rerank_score=float(s)) for h, s in zip(hits, scores, strict=True)]
        return sorted(scored, key=lambda h: -(h.rerank_score or 0.0))[:top_k]
