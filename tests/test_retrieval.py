"""Tests for retrieval backends, RRF fusion, and reranking (no network, no model downloads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.ingest import WordTokenizer, ingest_directory, load_chunks
from src.rerank import CrossEncoderReranker, NoopReranker
from src.retrieval import (
    HashingEmbedder,
    Hit,
    LocalBackend,
    SupabaseBackend,
    render_migration,
    rrf_fuse,
)

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture()
def chunks(tmp_path: Path) -> list[Any]:
    ingest_directory(FIXTURES, tmp_path, tokenizer=WordTokenizer())
    return load_chunks(tmp_path / "chunks.jsonl")


def test_rrf_fuse_scores_and_order() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [cid for cid, _ in fused][:2] == ["a", "c"]


def test_rrf_fuse_empty() -> None:
    assert rrf_fuse([[], []]) == []


def test_hashing_embedder_shape_and_norm() -> None:
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["roux flour butter", "danger zone"])
    assert vecs.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_local_backend_modes(chunks: list[Any]) -> None:
    backend = LocalBackend.build(chunks, HashingEmbedder(dim=256))
    bm25 = backend.search("roux flour butter", mode="bm25", limit=3)
    assert "roux" in bm25[0].text.lower()
    assert bm25[0].bm25_rank == 1 and bm25[0].dense_rank is None
    dense = backend.search("poultry internal temperature", mode="dense", limit=3)
    assert "poultry" in dense[0].text.lower()
    assert dense[0].dense_rank == 1 and dense[0].bm25_rank is None
    hybrid = backend.search("poultry internal temperature", limit=3)
    assert hybrid[0].bm25_rank is not None and hybrid[0].dense_rank is not None
    assert hybrid[0].score == pytest.approx(1 / 61 + 1 / 61)


def test_local_backend_rejects_unknown_mode(chunks: list[Any]) -> None:
    backend = LocalBackend.build(chunks, HashingEmbedder(dim=32))
    with pytest.raises(ValueError):
        backend.search("x", mode="sparse")  # type: ignore[arg-type]


def test_local_backend_save_load_roundtrip(chunks: list[Any], tmp_path: Path) -> None:
    emb = HashingEmbedder(dim=128)
    LocalBackend.build(chunks, emb).save(tmp_path / "idx")
    loaded = LocalBackend.load(tmp_path / "idx", emb)
    assert loaded.search("danger zone", limit=1)[0].chunk_id in {c.chunk_id for c in chunks}


def test_local_backend_dimension_mismatch(chunks: list[Any], tmp_path: Path) -> None:
    LocalBackend.build(chunks, HashingEmbedder(dim=128)).save(tmp_path / "idx")
    with pytest.raises(ValueError, match="dimension"):
        LocalBackend.load(tmp_path / "idx", HashingEmbedder(dim=64))


class _FakeRpc:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def execute(self) -> Any:
        return type("R", (), {"data": self._data})()


class _FakeSupabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.upserts: list[list[dict[str, Any]]] = []

    def rpc(self, name: str, params: dict[str, Any]) -> _FakeRpc:
        self.calls.append((name, params))
        row = lambda cid: {  # noqa: E731
            "chunk_id": cid,
            "content": f"text {cid}",
            "source_path": "p.txt",
            "title": "T",
            "source_type": "other",
        }
        if name == "match_chunks_dense":
            return _FakeRpc([row("a"), row("b")])
        return _FakeRpc([row("b"), row("c")])

    def table(self, name: str) -> Any:
        outer = self

        class _T:
            def upsert(self, rows: list[dict[str, Any]], **_: Any) -> _FakeRpc:
                outer.upserts.append(rows)
                return _FakeRpc(rows)

        return _T()


def test_supabase_backend_hybrid_uses_both_rpcs_and_fuses() -> None:
    client = _FakeSupabase()
    backend = SupabaseBackend(client, HashingEmbedder(dim=16))
    hits = backend.search("anything", limit=5)
    names = [n for n, _ in client.calls]
    assert names == ["match_chunks_dense", "match_chunks_bm25"]
    dense_params = client.calls[0][1]
    assert len(dense_params["query_embedding"]) == 16 and dense_params["match_count"] == 20
    assert client.calls[1][1]["query_text"] == "anything"
    assert [h.chunk_id for h in hits] == ["b", "a", "c"]  # b is rank 2 + rank 1


def test_supabase_upsert_chunks(chunks: list[Any]) -> None:
    client = _FakeSupabase()
    backend = SupabaseBackend(client, HashingEmbedder(dim=16))
    backend.upsert_chunks(chunks)
    row = client.upserts[0][0]
    assert {"chunk_id", "content", "embedding", "token_count"} <= set(row)
    assert len(row["embedding"]) == 16


def test_render_migration_sets_dimension() -> None:
    sql = render_migration(1536)
    assert "vector(1536)" in sql and "vector(384)" not in sql
    assert "match_chunks_dense" in sql and "match_chunks_bm25" in sql
    assert "using gin (content_tsv)" in sql


def test_committed_migration_matches_default_dim() -> None:
    path = next((Path(__file__).parents[1] / "supabase" / "migrations").glob("*.sql"))
    assert path.read_text() == render_migration(384)


class _FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(text)) for _, text in pairs]


def _hit(cid: str, text: str) -> Hit:
    return Hit(chunk_id=cid, text=text, source_path="p", title="t", source_type="other", score=0.0)


def test_cross_encoder_reranker_orders_and_truncates() -> None:
    rr = CrossEncoderReranker(model=_FakeCrossEncoder())
    out = rr.rerank("q", [_hit("a", "xx"), _hit("b", "xxxxx"), _hit("c", "xxx")], top_k=2)
    assert [h.chunk_id for h in out] == ["b", "c"]
    assert out[0].rerank_score == 5.0


def test_noop_reranker_truncates() -> None:
    out = NoopReranker().rerank("q", [_hit("a", "x"), _hit("b", "y")], top_k=1)
    assert [h.chunk_id for h in out] == ["a"]
