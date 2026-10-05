"""Hybrid retrieval: BM25 + dense vectors fused with reciprocal rank fusion (RRF).

Two interchangeable backends share one fusion path:

- ``LocalBackend``: rank-bm25 plus a numpy dense index. Default for CI, demo, offline use.
- ``SupabaseBackend``: Postgres tsvector + pgvector via RPC functions defined in
  ``supabase/migrations/``.

``mode`` exists for evals and debugging. Production callers use the default ``hybrid``
(design.md: never dense-only).
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
from rank_bm25 import BM25Okapi

from src.ingest import Chunk, load_chunks

Mode = Literal["hybrid", "bm25", "dense"]
Matrix = npt.NDArray[np.float32]
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    source_path: str
    title: str
    source_type: str
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> Matrix:
        """Return L2-normalized row vectors, shape (len(texts), dim)."""
        ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder. Offline smoke runs and tests only.

    Not a semantic model: it captures lexical overlap, so results from it must never
    be reported as dense-retrieval quality.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (default all-MiniLM-L6-v2, 384 dims)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> Matrix:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Ranks are 1-based; ties break by first appearance."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class RetrievalBackend(ABC):
    """Shared hybrid logic; subclasses supply the two ranked candidate lists."""

    rrf_k = 60
    candidates = 20

    @abstractmethod
    def bm25_search(self, query: str, n: int) -> list[Hit]: ...

    @abstractmethod
    def dense_search(self, query: str, n: int) -> list[Hit]: ...

    def search(self, query: str, mode: Mode = "hybrid", limit: int = 10) -> list[Hit]:
        if mode == "bm25":
            return [
                replace(h, bm25_rank=i, score=h.score)
                for i, h in enumerate(self.bm25_search(query, limit), 1)
            ]
        if mode == "dense":
            return [
                replace(h, dense_rank=i, score=h.score)
                for i, h in enumerate(self.dense_search(query, limit), 1)
            ]
        if mode != "hybrid":
            raise ValueError(f"unknown mode: {mode!r}")
        dense = self.dense_search(query, self.candidates)
        bm25 = self.bm25_search(query, self.candidates)
        by_id = {h.chunk_id: h for h in [*dense, *bm25]}
        bm25_rank = {h.chunk_id: i for i, h in enumerate(bm25, 1)}
        dense_rank = {h.chunk_id: i for i, h in enumerate(dense, 1)}
        fused = rrf_fuse([[h.chunk_id for h in dense], [h.chunk_id for h in bm25]], self.rrf_k)
        return [
            replace(
                by_id[cid],
                score=score,
                bm25_rank=bm25_rank.get(cid),
                dense_rank=dense_rank.get(cid),
            )
            for cid, score in fused[:limit]
        ]


def _hit_from_chunk(chunk: Chunk, score: float) -> Hit:
    return Hit(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source_path=chunk.source_path,
        title=chunk.title,
        source_type=chunk.source_type,
        score=score,
    )


class LocalBackend(RetrievalBackend):
    """BM25 (rank-bm25) + brute-force cosine over a numpy matrix."""

    def __init__(self, chunks: list[Chunk], vectors: Matrix, embedder: Embedder) -> None:
        if vectors.shape != (len(chunks), embedder.dim):
            raise ValueError(
                f"vector matrix {vectors.shape} does not match {len(chunks)} chunks "
                f"x dimension {embedder.dim}"
            )
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, batch: int = 64) -> LocalBackend:
        parts = [
            embedder.embed([c.text for c in chunks[i : i + batch]])
            for i in range(0, len(chunks), batch)
        ]
        vectors = np.vstack(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(chunks, vectors, embedder)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {
            "dim": self.embedder.dim,
            "model": getattr(self.embedder, "model_name", type(self.embedder).__name__),
        }
        (directory / "meta.json").write_text(json.dumps(meta))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.__dict__, ensure_ascii=False, sort_keys=True) + "\n")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> LocalBackend:
        meta = json.loads((directory / "meta.json").read_text())
        if meta["dim"] != embedder.dim:
            raise ValueError(
                f"index dimension {meta['dim']} != embedder dimension {embedder.dim}; "
                "rebuild the index"
            )
        vectors = np.load(directory / "vectors.npy")
        return cls(load_chunks(directory / "chunks.jsonl"), vectors, embedder)

    def bm25_search(self, query: str, n: int) -> list[Hit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:n]
        return [_hit_from_chunk(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def dense_search(self, query: str, n: int) -> list[Hit]:
        if not len(self.chunks):
            return []
        q = self.embedder.embed([query])[0]
        sims = self.vectors @ q
        order = np.argsort(-sims, kind="stable")[:n]
        return [_hit_from_chunk(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0]


class SupabaseBackend(RetrievalBackend):
    """pgvector + tsvector via the RPC functions in supabase/migrations/."""

    def __init__(self, client: Any, embedder: Embedder) -> None:
        self.client = client
        self.embedder = embedder

    @staticmethod
    def _to_hits(rows: list[dict[str, Any]]) -> list[Hit]:
        return [
            Hit(
                chunk_id=r["chunk_id"],
                text=r["content"],
                source_path=r.get("source_path", ""),
                title=r.get("title", ""),
                source_type=r.get("source_type", "other"),
                score=float(r.get("score", 0.0)),
            )
            for r in rows
        ]

    def dense_search(self, query: str, n: int) -> list[Hit]:
        emb = self.embedder.embed([query])[0].tolist()
        res = self.client.rpc(
            "match_chunks_dense", {"query_embedding": emb, "match_count": n}
        ).execute()
        return self._to_hits(res.data)

    def bm25_search(self, query: str, n: int) -> list[Hit]:
        res = self.client.rpc(
            "match_chunks_bm25", {"query_text": query, "match_count": n}
        ).execute()
        return self._to_hits(res.data)

    def upsert_chunks(self, chunks: list[Chunk], batch: int = 100) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            vecs = self.embedder.embed([c.text for c in part])
            rows = [
                {
                    "chunk_id": c.chunk_id,
                    "source_path": c.source_path,
                    "title": c.title,
                    "source_type": c.source_type,
                    "chunk_index": c.chunk_index,
                    "content": c.text,
                    "token_count": c.token_count,
                    "embedding": v.tolist(),
                }
                for c, v in zip(part, vecs, strict=True)
            ]
            self.client.table("chunks").upsert(rows, on_conflict="chunk_id").execute()


_MIGRATION_TEMPLATE = Path(__file__).with_name("schema.sql.tmpl")


def render_migration(dim: int) -> str:
    """Render the Supabase schema for an embedding dimension (384 for MiniLM, 1536 for OpenAI)."""
    return _MIGRATION_TEMPLATE.read_text(encoding="utf-8").replace("{{EMBEDDING_DIM}}", str(dim))
