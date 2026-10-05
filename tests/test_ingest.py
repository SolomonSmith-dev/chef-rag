"""Tests for ingestion: loading, cleaning, chunking, and JSONL output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest import (
    Chunk,
    WordTokenizer,
    chunk_document,
    ingest_directory,
    load_chunks,
    load_documents,
    strip_gutenberg,
)

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


def test_strip_gutenberg_removes_boilerplate() -> None:
    raw = (FIXTURES / "fixture_escoffier.txt").read_text(encoding="utf-8")
    body = strip_gutenberg(raw)
    assert "START OF THE PROJECT GUTENBERG" not in body
    assert "END OF THE PROJECT GUTENBERG" not in body
    assert "Licensing text" not in body
    assert "mother sauces" in body.lower()


def test_strip_gutenberg_passthrough_without_markers() -> None:
    assert strip_gutenberg("plain text") == "plain text"


def test_load_documents_metadata() -> None:
    docs = {d.source_path: d for d in load_documents(FIXTURES)}
    assert set(docs) == {
        "fixture_escoffier.txt",
        "fixture_fda_foodcode.txt",
        "fixture_note.md",
    }
    assert docs["fixture_escoffier.txt"].source_type == "gutenberg"
    assert docs["fixture_fda_foodcode.txt"].source_type == "fda"
    assert docs["fixture_note.md"].source_type == "original"
    assert docs["fixture_note.md"].title == "Fixture Original Note"
    assert "---" not in docs["fixture_note.md"].text


def test_chunk_sizes_and_overlap() -> None:
    tok = WordTokenizer()
    text = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_document("doc.txt", text, tok, chunk_size=500, overlap=50)
    assert len(chunks) >= 3
    for c in chunks:
        assert tok.count(c) <= 500
    # consecutive chunks share a 50-token tail/head overlap
    tail = chunks[0].split()[-50:]
    assert chunks[1].split()[:50] == tail


def test_chunk_prefers_paragraph_boundaries() -> None:
    tok = WordTokenizer()
    p1 = " ".join(["alpha"] * 300)
    p2 = " ".join(["beta"] * 300)
    chunks = chunk_document("d.txt", f"{p1}\n\n{p2}", tok, chunk_size=400, overlap=0)
    assert chunks[0] == p1
    assert chunks[1] == p2


def test_chunk_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_document("d", "x", WordTokenizer(), chunk_size=10, overlap=10)


def test_ingest_writes_stable_jsonl(tmp_path: Path) -> None:
    out1 = ingest_directory(FIXTURES, tmp_path / "a", tokenizer=WordTokenizer())
    out2 = ingest_directory(FIXTURES, tmp_path / "b", tokenizer=WordTokenizer())
    assert (tmp_path / "a" / "chunks.jsonl").read_text() == (
        tmp_path / "b" / "chunks.jsonl"
    ).read_text()
    assert out1 == out2 > 0
    chunks = load_chunks(tmp_path / "a" / "chunks.jsonl")
    assert all(isinstance(c, Chunk) for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    first = json.loads((tmp_path / "a" / "chunks.jsonl").read_text().splitlines()[0])
    assert {
        "chunk_id",
        "source_path",
        "title",
        "source_type",
        "chunk_index",
        "text",
        "token_count",
    } <= set(first)


def test_chunk_id_stable_when_unrelated_doc_added(tmp_path: Path) -> None:
    ingest_directory(FIXTURES, tmp_path / "a", tokenizer=WordTokenizer())
    extra = tmp_path / "corpus"
    extra.mkdir()
    for p in FIXTURES.iterdir():
        (extra / p.name).write_text(p.read_text())
    (extra / "zzz_extra.txt").write_text("extra document about braising")
    ingest_directory(extra, tmp_path / "b", tokenizer=WordTokenizer())
    a = {c.chunk_id for c in load_chunks(tmp_path / "a" / "chunks.jsonl")}
    b = {c.chunk_id for c in load_chunks(tmp_path / "b" / "chunks.jsonl")}
    assert a < b
