# Corpus runbook (run on your Mac)

The build sandbox blocks gutenberg.org, fda.gov and fsis.usda.gov, so the corpus is
fetched by you. Everything else is tested against `tests/fixtures/corpus/`.

Run from the repo root. Commands use full paths for `grep`, `ls` and `cp -f`.

## 1. Sync

```bash
git checkout feat/ingest && git pull && uv sync --all-extras
```
Expect: `Resolved ... packages` with no error.

## 2. Download

```bash
uv run python scripts/fetch_corpus.py
```
Expect: `OK: 6 sources` and exit code 0 (`echo $?` prints `0`).
If you see `FAIL <name>: ...`, a Gutenberg id or agency URL moved. Fix the `url` in
`scripts/fetch_corpus.py` `SOURCES` and `data/SOURCES.md`, then rerun. A
`expected phrase ... not found` failure means the URL returned the wrong document.

## 3. Check the files

```bash
/bin/ls -la data/raw/
```
Expect: 6 `.txt` files plus `manifest.json`. The two Gutenberg books should each
be hundreds of KB. The FDA Food Code should be larger than 1 MB of text.

```bash
/usr/bin/grep -c "PROJECT GUTENBERG" data/raw/escoffier_guide_to_modern_cookery.txt
```
Expect: `0`.

## 4. Done check for the download (can fail)

```bash
uv run python scripts/fetch_corpus.py --verify; echo "exit=$?"
```
Expect: `OK: 6 sources` and `exit=0`. Any `FAIL` line and `exit=1` means not done.

## 5. Ingest

```bash
uv run python -m src.cli ingest --source data/raw/ --out data/processed/
```
Expect: `wrote N chunks to data/processed/chunks.jsonl` with N in the thousands.

## 6. Done check for ingest (can fail)

```bash
for t in gutenberg fda usda; do
  n=$(/usr/bin/grep -c "\"source_type\": \"$t\"" data/processed/chunks.jsonl)
  echo "$t=$n"; [ "$n" -gt 0 ] || { echo "FAIL: no $t chunks"; exit 1; }
done
```
Expect: three lines, each count greater than 0, and no `FAIL`.

## 7. Record retrieval dates

```bash
/bin/cp -f data/SOURCES.md data/SOURCES.md.bak
```
Then replace each "not yet retrieved" in `data/SOURCES.md` with the matching
`retrieved` date from `data/raw/manifest.json`, and delete the `.bak` file.
