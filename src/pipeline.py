"""Wiring: build embedder, backend and reranker from settings, and run retrieve + rerank."""

from __future__ import annotations

from pathlib import Path

from src.config import RetrievalSettings
from src.ingest import load_chunks
from src.rerank import CrossEncoderReranker, NoopReranker, Reranker
from src.retrieval import (
    Embedder,
    HashingEmbedder,
    Hit,
    LocalBackend,
    Mode,
    RetrievalBackend,
    SentenceTransformerEmbedder,
    SupabaseBackend,
)


def build_embedder(name: str, cfg: RetrievalSettings) -> Embedder:
    """``minilm`` is the real local model; ``hash`` is an offline stand-in (not semantic)."""
    if name == "hash":
        return HashingEmbedder(cfg.embedding_dim)
    emb = SentenceTransformerEmbedder(cfg.local_embedding_model)
    if emb.dim != cfg.embedding_dim:
        raise ValueError(
            f"{cfg.local_embedding_model} outputs {emb.dim} dims "
            f"but EMBEDDING_DIM={cfg.embedding_dim}"
        )
    return emb


def build_reranker(name: str, cfg: RetrievalSettings) -> Reranker:
    return NoopReranker() if name == "none" else CrossEncoderReranker(cfg.rerank_model)


def build_index(cfg: RetrievalSettings, embedder: Embedder) -> int:
    """Embed data/processed chunks and save the local index. Returns the chunk count."""
    chunks = load_chunks(Path(cfg.chunks_path))
    LocalBackend.build(chunks, embedder).save(Path(cfg.index_dir))
    return len(chunks)


def build_backend(name: str, cfg: RetrievalSettings, embedder: Embedder) -> RetrievalBackend:
    if name == "local":
        return LocalBackend.load(Path(cfg.index_dir), embedder)
    if name == "supabase":
        from src.config import load_settings
        from supabase import create_client

        s = load_settings()
        return SupabaseBackend(create_client(s.supabase_url, s.supabase_key), embedder)
    raise ValueError(f"unknown backend: {name!r}")


def retrieve(
    query: str,
    backend: RetrievalBackend,
    reranker: Reranker,
    cfg: RetrievalSettings,
    mode: Mode = "hybrid",
    k: int | None = None,
) -> tuple[list[Hit], list[Hit]]:
    """Return (candidates before rerank, final top-k)."""
    candidates = backend.search(query, mode=mode, limit=cfg.fuse_top_n)
    return candidates, reranker.rerank(query, candidates, k or cfg.final_top_k)
