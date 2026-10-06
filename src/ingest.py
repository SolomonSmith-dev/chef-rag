"""Corpus ingestion: load, clean, chunk, and write JSONL.

Chunking follows docs/design.md: 500-token target, 50-token overlap, recursive
character splitter measured in ``cl100k_base`` tokens.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

SOURCE_TYPES = {"gutenberg", "fda", "usda", "original", "other"}
TEXT_SUFFIXES = {".txt", ".md"}
SEPARATORS = ["\n\n", "\n", ". ", " "]

_GUTENBERG_START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.M)
_GUTENBERG_END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.M)


class Tokenizer(Protocol):
    """Token counting plus an overlap helper, so tests can run without BPE downloads."""

    def count(self, text: str) -> int: ...

    def tail(self, text: str, n: int) -> str:
        """Return roughly the last ``n`` tokens of ``text`` starting at a word boundary."""
        ...


class TiktokenTokenizer:
    """``cl100k_base`` tokenizer (the production default)."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def tail(self, text: str, n: int) -> str:
        ids = self._enc.encode(text)
        if len(ids) <= n:
            return text
        piece = self._enc.decode(ids[-n:])
        # drop a possibly split leading word so overlap starts on a boundary
        if not piece[:1].isspace() and " " in piece:
            piece = piece.split(" ", 1)[1]
        return piece.lstrip("�")


class WordTokenizer:
    """Whitespace tokenizer used by tests and offline smoke runs. One word = one token."""

    def count(self, text: str) -> int:
        return len(text.split())

    def tail(self, text: str, n: int) -> str:
        return " ".join(text.split()[-n:])


@dataclass(frozen=True)
class Document:
    source_path: str
    title: str
    source_type: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str
    title: str
    source_type: str
    chunk_index: int
    text: str
    token_count: int


def strip_gutenberg(raw: str) -> str:
    """Remove Project Gutenberg header and license footer when the markers exist."""
    start = _GUTENBERG_START.search(raw)
    if start:
        raw = raw[start.end() :]
    end = _GUTENBERG_END.search(raw)
    if end:
        raw = raw[: end.start()]
    return raw.strip() if (start or end) else raw


def clean_text(text: str) -> str:
    """Normalize newlines and whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = re.match(r"\A---\n(.*?)\n---\n?", raw, re.S)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return meta, raw[match.end() :]


def load_documents(source_dir: Path) -> list[Document]:
    """Load every .txt/.md under ``source_dir`` (README.md files are skipped)."""
    docs: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name.lower() == "readme.md":
            continue
        meta, body = _split_front_matter(path.read_text(encoding="utf-8"))
        source_type = meta.get("source_type", "other")
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"{path}: unknown source_type {source_type!r}")
        body = clean_text(strip_gutenberg(body))
        if not body:
            continue
        docs.append(
            Document(
                source_path=path.relative_to(source_dir).as_posix(),
                title=meta.get("title", path.stem.replace("_", " ")),
                source_type=source_type,
                text=body,
            )
        )
    return docs


def _atoms(text: str, tok: Tokenizer, limit: int, seps: list[str]) -> list[str]:
    """Recursively split ``text`` until every piece fits in ``limit`` tokens."""
    if tok.count(text) <= limit:
        return [text]
    if not seps:
        words = text.split(" ")
        mid = max(1, len(words) // 2)
        if len(words) == 1:
            return [text]  # unsplittable single token run; accept oversize
        return _atoms(" ".join(words[:mid]), tok, limit, []) + _atoms(
            " ".join(words[mid:]), tok, limit, []
        )
    sep, rest = seps[0], seps[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return _atoms(text, tok, limit, rest)
    out: list[str] = []
    for i, part in enumerate(parts):
        piece = part + "." if sep == ". " and i < len(parts) - 1 else part
        out.extend(_atoms(piece, tok, limit, rest))
    return out


def chunk_document(
    name: str, text: str, tok: Tokenizer, chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` tokens with ``overlap`` carry-over.

    The body of each chunk is built from paragraph/sentence/word atoms up to
    ``chunk_size - overlap`` tokens; every chunk after the first is prefixed with
    the last ``overlap`` tokens of its predecessor.
    """
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"{name}: overlap must be in [0, chunk_size)")
    budget = chunk_size - overlap if overlap else chunk_size
    atoms = _atoms(text, tok, budget, SEPARATORS)
    bodies: list[str] = []
    current = ""
    for atom in atoms:
        joiner = "\n\n" if "\n\n" in text and current else " "
        candidate = f"{current}{joiner}{atom}" if current else atom
        if current and tok.count(candidate) > budget:
            bodies.append(current)
            current = atom
        else:
            current = candidate
    if current:
        bodies.append(current)
    if not overlap:
        return bodies
    chunks = [bodies[0]] if bodies else []
    for prev, body in zip(bodies, bodies[1:], strict=False):
        chunks.append(f"{tok.tail(prev, overlap)} {body}")
    return chunks


def _chunk_id(source_path: str, index: int, text: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(source_path).stem.lower()).strip("-")
    digest = hashlib.sha1(f"{source_path}\0{index}\0{text}".encode()).hexdigest()[:8]
    return f"{stem}-{index:04d}-{digest}"


def build_chunks(
    docs: list[Document], tok: Tokenizer, chunk_size: int = 500, overlap: int = 50
) -> Iterator[Chunk]:
    for doc in docs:
        for i, text in enumerate(
            chunk_document(doc.source_path, doc.text, tok, chunk_size, overlap)
        ):
            yield Chunk(
                chunk_id=_chunk_id(doc.source_path, i, text),
                source_path=doc.source_path,
                title=doc.title,
                source_type=doc.source_type,
                chunk_index=i,
                text=text,
                token_count=tok.count(text),
            )


def ingest_directory(
    source_dir: Path,
    out_dir: Path,
    tokenizer: Tokenizer | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> int:
    """Chunk every document in ``source_dir`` into ``out_dir/chunks.jsonl``. Returns chunk count."""
    tok = tokenizer or TiktokenTokenizer()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(build_chunks(load_documents(source_dir), tok, chunk_size, overlap))
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) + "\n")
    return len(chunks)


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]
