"""Tests for golden eval dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "evals" / "golden.jsonl"
REQUIRED_FIELDS = {"id", "category", "question", "reference_answer", "tags"}
VALID_CATEGORIES = {"technique", "safety", "management", "precision"}


def load_golden_records() -> list[dict[str, object]]:
    """Load all golden eval records from disk."""
    records: list[dict[str, object]] = []
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"line {line_number}: invalid JSON") from exc
            records.append(record)
    return records


def test_golden_file_exists() -> None:
    assert GOLDEN_PATH.is_file()


def test_golden_has_minimum_pairs() -> None:
    records = load_golden_records()
    assert len(records) >= 5


def test_golden_record_schema() -> None:
    records = load_golden_records()
    ids: set[str] = set()
    for record in records:
        assert REQUIRED_FIELDS.issubset(record.keys())
        record_id = str(record["id"])
        assert record_id not in ids, f"duplicate id: {record_id}"
        ids.add(record_id)
        assert str(record["category"]) in VALID_CATEGORIES
        assert str(record["question"]).strip()
        assert str(record["reference_answer"]).strip()
        tags = record["tags"]
        assert isinstance(tags, list)
        assert tags


@pytest.mark.parametrize(
    "category",
    sorted(VALID_CATEGORIES),
)
def test_golden_covers_each_category(category: str) -> None:
    records = load_golden_records()
    categories = {str(record["category"]) for record in records}
    assert category in categories
