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
        "--tokenizer",
        choices=["cl100k", "words"],
        default="cl100k",
        help="cl100k is the spec; words is an offline stand-in for smoke runs and tests",
    )

    query_parser = subparsers.add_parser("query", help="Query the culinary knowledge base")
    query_parser.add_argument("question", help="Natural language question")

    return parser


def _ingest(source: str, out: str, tokenizer: str) -> int:
    from pathlib import Path

    from src.ingest import TiktokenTokenizer, WordTokenizer, ingest_directory

    if not Path(source).is_dir():
        print(f"source directory not found: {source}", file=sys.stderr)
        return 2
    tok = WordTokenizer() if tokenizer == "words" else TiktokenTokenizer()
    count = ingest_directory(Path(source), Path(out), tokenizer=tok)
    print(f"wrote {count} chunks to {Path(out) / 'chunks.jsonl'}")
    return 0 if count else 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _ingest(args.source, args.out, args.tokenizer)

    if args.command == "query":
        print(f"query not implemented yet (question={args.question!r})", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
