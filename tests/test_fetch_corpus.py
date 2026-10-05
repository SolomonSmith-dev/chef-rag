"""Tests for scripts/fetch_corpus.py using an injected fetcher (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_corpus.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_corpus", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


fc = _load()

GUTENBERG = (
    b"Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
    b"Roux is flour and butter cooked together.\n"
    b"*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense junk\n"
)
HTML = (
    b"<html><head><title>t</title><script>var x=1;</script></head><body>"
    b"<nav>menu</nav><h1>Danger Zone</h1><p>Keep food out of 40 F to 140 F.</p></body></html>"
)


def test_html_to_text_drops_scripts_and_keeps_text() -> None:
    text = fc.html_to_text(HTML.decode())
    assert "Danger Zone" in text and "40 F to 140 F" in text
    assert "var x" not in text


def test_fetch_all_writes_files_manifest_and_strips(tmp_path: Path) -> None:
    sources = [
        fc.Source("roux_book", "Roux Book", "gutenberg", "http://x/g", "txt", "Roux is flour"),
        fc.Source("zone", "Zone", "usda", "http://x/h", "html", "Danger Zone"),
    ]
    payloads = {"http://x/g": GUTENBERG, "http://x/h": HTML}
    failures = fc.fetch_all(
        sources, tmp_path, fetcher=lambda url: payloads[url], today="2026-01-02"
    )
    assert failures == []
    body = (tmp_path / "roux_book.txt").read_text()
    assert "source_type: gutenberg" in body
    assert "PROJECT GUTENBERG" not in body and "license junk" not in body
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["roux_book"]["retrieved"] == "2026-01-02"
    assert len(manifest["roux_book"]["sha256"]) == 64
    assert fc.verify(sources, tmp_path, min_chars=10) == []


def test_fetch_all_reports_wrong_document(tmp_path: Path) -> None:
    src = fc.Source("a", "A", "gutenberg", "http://x/g", "txt", "Escoffier")
    failures = fc.fetch_all([src], tmp_path, fetcher=lambda url: GUTENBERG, today="2026-01-02")
    assert failures and "expected phrase" in failures[0]
    assert not (tmp_path / "a.txt").exists()


def test_fetch_all_reports_network_error(tmp_path: Path) -> None:
    def boom(url: str) -> bytes:
        raise OSError("blocked")

    src = fc.Source("a", "A", "usda", "http://x/h", "html", "Zone")
    failures = fc.fetch_all([src], tmp_path, fetcher=boom, today="2026-01-02")
    assert failures and "blocked" in failures[0]


def test_verify_fails_on_missing_and_on_boilerplate(tmp_path: Path) -> None:
    src = fc.Source("a", "A", "gutenberg", "u", "txt", "Roux")
    assert fc.verify([src], tmp_path)  # missing file
    (tmp_path / "a.txt").write_text("---\ntitle: A\n---\nRoux *** START OF THE PROJECT GUTENBERG")
    assert any("boilerplate" in f for f in fc.verify([src], tmp_path))


def test_default_sources_declare_license_basis() -> None:
    assert len(fc.SOURCES) >= 5
    for s in fc.SOURCES:
        assert s.url.startswith("https://") and s.expect
    with pytest.raises(SystemExit):
        fc.main(["--bogus"])
