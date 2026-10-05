"""Download the public-domain corpus into data/raw/ and strip boilerplate.

Usage:
    uv run python scripts/fetch_corpus.py            # download
    uv run python scripts/fetch_corpus.py --verify   # done check, exits non-zero on failure

Every document is written as ``<name>.txt`` with a small front matter block
(title, source_type, url) that ``src.ingest`` reads. ``data/raw/manifest.json``
records the retrieval date and sha256 of each file.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ingest import clean_text, strip_gutenberg  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    source_type: str  # gutenberg | fda | usda
    url: str
    fmt: str  # txt | html | pdf
    expect: str  # phrase that must appear in the cleaned text (wrong-document guard)


# Project Gutenberg ids and agency URLs could not be checked from the build
# sandbox (egress blocked). The `expect` phrase makes a wrong id or moved page fail
# loudly instead of ingesting the wrong document.
SOURCES: list[Source] = [
    Source(
        "escoffier_guide_to_modern_cookery",
        "A Guide to Modern Cookery (Escoffier, 1907)",
        "gutenberg",
        "https://www.gutenberg.org/cache/epub/47703/pg47703.txt",
        "txt",
        "Escoffier",
    ),
    Source(
        "farmer_boston_cooking_school_cook_book",
        "The Boston Cooking-School Cook Book (Farmer, 1896)",
        "gutenberg",
        "https://www.gutenberg.org/cache/epub/65/pg65.txt",
        "txt",
        "Boston Cooking-School",
    ),
    Source(
        "beeton_household_management",
        "Mrs Beeton's Book of Household Management (1861)",
        "gutenberg",
        "https://www.gutenberg.org/cache/epub/10136/pg10136.txt",
        "txt",
        "Household Management",
    ),
    Source(
        "fda_food_code_2022",
        "FDA Food Code 2022",
        "fda",
        "https://www.fda.gov/media/164194/download",
        "pdf",
        "Food Code",
    ),
    Source(
        "fsis_safe_minimum_internal_temperature_chart",
        "USDA FSIS Safe Minimum Internal Temperature Chart",
        "usda",
        "https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/safe-temperature-chart",
        "html",
        "165",
    ),
    Source(
        "fsis_danger_zone",
        "USDA FSIS Danger Zone (40 F - 140 F)",
        "usda",
        "https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/danger-zone-40f-140f",
        "html",
        "140",
    ),
]

Fetcher = Callable[[str], bytes]


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "nav", "header", "footer", "noscript"}
    _BLOCK = {"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "section", "table"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n\n" if tag.startswith(("h", "p")) else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    return html.unescape("".join(parser.parts))


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def http_fetch(url: str) -> bytes:
    import httpx

    resp = httpx.get(
        url, follow_redirects=True, timeout=120.0, headers={"User-Agent": "chef-rag/0.1"}
    )
    resp.raise_for_status()
    return resp.content


def _extract(src: Source, data: bytes) -> str:
    if src.fmt == "pdf":
        return pdf_to_text(data)
    text = data.decode("utf-8", errors="replace")
    return html_to_text(text) if src.fmt == "html" else text


def fetch_all(
    sources: list[Source],
    out_dir: Path,
    fetcher: Fetcher = http_fetch,
    today: str | None = None,
) -> list[str]:
    """Fetch each source. Returns a list of failure messages (empty on success)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, dict[str, str]] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    failures: list[str] = []
    for src in sources:
        try:
            raw = fetcher(src.url)
            text = clean_text(strip_gutenberg(_extract(src, raw)))
        except Exception as exc:  # noqa: BLE001 - report and continue with other sources
            failures.append(f"{src.name}: fetch failed: {exc}")
            continue
        if src.expect.lower() not in text.lower():
            failures.append(
                f"{src.name}: expected phrase {src.expect!r} not found; wrong document?"
            )
            continue
        head = f"---\ntitle: {src.title}\nsource_type: {src.source_type}\nurl: {src.url}\n---\n"
        content = f"{head}{text}\n"
        (out_dir / f"{src.name}.txt").write_text(content, encoding="utf-8")
        manifest[src.name] = {
            "url": src.url,
            "retrieved": today or date.today().isoformat(),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return failures


def verify(sources: list[Source], out_dir: Path, min_chars: int = 2000) -> list[str]:
    """Done check: every file exists, is non-trivial, has the expected phrase, no boilerplate."""
    problems: list[str] = []
    for src in sources:
        path = out_dir / f"{src.name}.txt"
        if not path.is_file():
            problems.append(f"{src.name}: missing {path}")
            continue
        body = path.read_text(encoding="utf-8")
        if re.search(r"\*{3}\s*(START|END) OF .*GUTENBERG", body, re.I):
            problems.append(f"{src.name}: Gutenberg boilerplate still present")
        if src.expect.lower() not in body.lower():
            problems.append(f"{src.name}: expected phrase {src.expect!r} missing")
        if len(body) < min_chars and src.fmt != "html":
            problems.append(f"{src.name}: suspiciously short ({len(body)} chars)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="run the done check only")
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)
    problems = verify(SOURCES, args.out) if args.verify else fetch_all(SOURCES, args.out)
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    if not problems:
        print(f"OK: {len(SOURCES)} sources", file=sys.stderr if args.verify else sys.stdout)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
