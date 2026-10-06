"""Assemble the Hugging Face Space build directory (Docker SDK).

    uv run python scripts/prepare_space.py --out build/hf-space

Copies only what the Dockerfile needs. Never copies .env files. The corpus itself is
downloaded during the Docker build, not copied. Your own notes in
data/raw/original/*.md are included if present, which PUBLISHES them with the Space.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FILES = ["pyproject.toml", "uv.lock", "LICENSE", "data/SOURCES.md"]
DIRS = ["src", "scripts"]
SKIP = {"__pycache__", ".env", ".pytest_cache", ".mypy_cache"}


def _ignore(_: str, names: list[str]) -> set[str]:
    return {n for n in names if n in SKIP or n.endswith(".pyc") or n.startswith(".env")}


def prepare(out: Path, repo: Path = REPO_ROOT) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copied: list[str] = []
    for rel in FILES:
        src = repo / rel
        if src.is_file():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    for rel in DIRS:
        shutil.copytree(repo / rel, out / rel, ignore=_ignore)
        copied.extend(p.relative_to(out).as_posix() for p in (out / rel).rglob("*") if p.is_file())
    for name in ("Dockerfile", "README.md"):
        shutil.copy2(repo / "deploy" / "hf-space" / name, out / name)
        copied.append(name)
    (out / "data" / "raw" / "original").mkdir(parents=True, exist_ok=True)
    for note in sorted((repo / "data" / "raw" / "original").glob("*.md")):
        if note.name.lower() != "readme.md":
            shutil.copy2(note, out / "data" / "raw" / "original" / note.name)
            copied.append(f"data/raw/original/{note.name}")
    (out / "data" / "raw" / "original" / ".keep").write_text("")
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "hf-space")
    args = ap.parse_args(argv)
    copied = prepare(args.out)
    missing = [
        n for n in ("Dockerfile", "README.md", "pyproject.toml", "src/demo.py") if n not in copied
    ]
    if missing:
        print(f"FAIL missing from space dir: {missing}", file=sys.stderr)
        return 1
    notes = [c for c in copied if c.startswith("data/raw/original/")]
    print(f"prepared {len(copied)} files in {args.out} ({len(notes)} original notes included)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
