# Original notes (yours only)

Drop your own kitchen notes here as one Markdown file per topic. Nothing in this
folder is written by tooling. Only documents you author belong here, and only they
may carry `source_type: original`.

## File format

```markdown
---
title: Pre-shift meeting for an 8-person line
source_type: original
---
Plain Markdown body. Paragraphs separated by blank lines.
Headings are fine. Keep one technique or procedure per file when you can.
```

Rules:

- Front matter is required: `title` and `source_type: original`.
- File name: lowercase with underscores, e.g. `pre_shift_meeting.md`.
- Files named `README.md` are skipped by ingest.
- This folder's contents are gitignored by default. To publish a note, run
  `git add -f data/raw/original/<file>.md` (the README is already tracked).

## Check it works

```bash
uv run python -m src.cli ingest --source data/raw/ --out data/processed/
grep -c '"source_type": "original"' data/processed/chunks.jsonl
```

The count must be greater than 0 once you have added a note.
