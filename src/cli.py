"""CLI entry point for chef-rag."""

from __future__ import annotations

import argparse
import sys

from src import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="chef-rag",
        description="Production-grade RAG for professional culinary knowledge.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a source directory")
    ingest_parser.add_argument("--source", required=True, help="Path to raw documents")
    ingest_parser.add_argument("--out", default="data/processed", help="Output directory")
    ingest_parser.add_argument(
        "--index", action="store_true", help="also embed chunks and build the local index"
    )
    ingest_parser.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    ingest_parser.add_argument(
        "--tokenizer",
        choices=["cl100k", "words"],
        default="cl100k",
        help="cl100k is the spec; words is an offline stand-in for smoke runs and tests",
    )

    query_parser = subparsers.add_parser("query", help="Query the culinary knowledge base")
    query_parser.add_argument("question", help="Natural language question")
    query_parser.add_argument("--backend", choices=["local", "supabase"], default="local")
    query_parser.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    query_parser.add_argument("--reranker", choices=["bge", "none"], default="bge")
    query_parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25", "dense"],
        default="hybrid",
        help="eval/debug only; production is always hybrid",
    )
    query_parser.add_argument("--k", type=int, default=5, help="results after rerank")
    query_parser.add_argument("--show-scores", action="store_true")

    return parser


def _ingest(source: str, out: str, tokenizer: str, index: bool, embedder: str) -> int:
    from pathlib import Path

    from src.ingest import TiktokenTokenizer, WordTokenizer, ingest_directory

    if not Path(source).is_dir():
        print(f"source directory not found: {source}", file=sys.stderr)
        return 2
    tok = WordTokenizer() if tokenizer == "words" else TiktokenTokenizer()
    count = ingest_directory(Path(source), Path(out), tokenizer=tok)
    print(f"wrote {count} chunks to {Path(out) / 'chunks.jsonl'}")
    if count and index:
        from src.config import load_retrieval_settings
        from src.pipeline import build_embedder, build_index

        cfg = load_retrieval_settings().model_copy(
            update={
                "chunks_path": str(Path(out) / "chunks.jsonl"),
                "index_dir": str(Path(out) / "index"),
            }
        )
        n = build_index(cfg, build_embedder(embedder, cfg))
        print(f"indexed {n} chunks (dim={cfg.embedding_dim}) in {cfg.index_dir}")
    return 0 if count else 1


def _query(args: argparse.Namespace) -> int:
    from src.config import load_retrieval_settings
    from src.pipeline import build_backend, build_embedder, build_reranker, retrieve

    cfg = load_retrieval_settings()
    try:
        backend = build_backend(args.backend, cfg, build_embedder(args.embedder, cfg))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc} (run `chef-rag ingest --index` first?)", file=sys.stderr)
        return 2
    candidates, hits = retrieve(
        args.question, backend, build_reranker(args.reranker, cfg), cfg, args.mode, args.k
    )
    if not hits:
        print("no results")
        return 1
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.chunk_id}  ({h.title})")
        if args.show_scores:
            print(
                f"    rrf={h.score:.4f} bm25_rank={h.bm25_rank} dense_rank={h.dense_rank} "
                f"rerank={h.rerank_score}"
            )
        print(f"    {h.text[:300].strip()}")
    if args.show_scores:
        print(f"candidates before rerank: {len(candidates)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _ingest(args.source, args.out, args.tokenizer, args.index, args.embedder)

    if args.command == "query":
        return _query(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
