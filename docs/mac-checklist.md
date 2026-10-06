# Mac checklist: corpus, first eval, deploy

One pass, in order. Every step has a check that can fail. Details live in
`docs/corpus-runbook.md`, `docs/eval-runbook.md` and `docs/deploy-hf.md`.
Commands use full paths for `grep`, `ls` and `cp -f`.

## 0. Before you run anything

1. `git checkout main && git pull && uv sync --all-extras`
2. Review `evals/golden.jsonl`. Fix these first:
   - `safety-002`: Food Code 2022 says 165F instantaneous, not 15 seconds.
   - `safety-003`: hot hold is 135F; the 4-hour rule is time as a control.
   - `precision-001`: chef pull temp 130-135F vs FDA/USDA 145F. Pick one.
3. Skim every `needs_review: true` question. Delete or fix what you disagree with.

## 1. Corpus

```bash
uv run python scripts/fetch_corpus.py && uv run python scripts/fetch_corpus.py --verify
```
Pass: `OK: 6 sources`. Fail: a `FAIL <name>` line; fix that URL in the script.

## 2. Your notes

Add Markdown files to `data/raw/original/` (format in its README).

```bash
uv run python -m src.cli ingest --source data/raw/ --out data/processed/ --index
/usr/bin/grep -c '"source_type": "original"' data/processed/chunks.jsonl
```
Pass: the count is above 0.

## 3. Offline ablation (free)

```bash
uv run python evals/run_evals.py && uv run python scripts/check_results.py
```
Pass: 4-row table, then `OK`. Open the JSON: read `unresolved` and `gate_sweep`.
Set `export MIN_RERANK_SCORE=<pick from gate_sweep>`.

## 4. First live run, small and cheap

```bash
export OPENROUTER_API_KEY=...        # shell only, never .env
export GENERATION_MODEL=anthropic/claude-haiku-4.5
uv run python evals/run_evals.py --live --gen-configs hybrid_rerank --limit 20
```
Pass: `projected live cost` under `remaining`, and a result with Ragas columns.
The run aborts before spending if the projection exceeds the cap.

Then the full headline pair:

```bash
uv run python evals/run_evals.py --live --gen-configs dense,hybrid_rerank
uv run python scripts/check_results.py --live
```
Pass: `OK`.

## 5. Record

```bash
uv run python scripts/render_results.py
git checkout -b results/v1-baseline && git add evals/results README.md ROADMAP.md
```
Commit, push, open a PR.

## 6. Deploy

Follow `docs/deploy-hf.md` (assemble, push the Space, add the secret in the HF UI,
run `scripts/check_demo.py`). Then replace the README demo placeholder.
