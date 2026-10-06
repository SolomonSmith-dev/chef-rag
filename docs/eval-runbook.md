# Eval runbook (run on your Mac)

The build sandbox could not reach Hugging Face, OpenRouter, Gutenberg, FDA or FSIS
and had no API keys, so no real eval number exists yet. This produces the baseline.
Commands use full paths for `grep`, `ls` and `cp -f`.

## 1. Corpus and index

Finish `docs/corpus-runbook.md` first, then:

```bash
uv run python -m src.cli ingest --source data/raw/ --out data/processed/ --index
```
Expect: `wrote N chunks ...` then `indexed N chunks (dim=384) in data/processed/index`.
The first run downloads all-MiniLM-L6-v2 (about 90 MB).

```bash
/bin/ls data/processed/index
```
Expect: `chunks.jsonl  meta.json  vectors.npy`.

## 2. Offline ablation (no LLM calls, no cost)

```bash
uv run python evals/run_evals.py
```
Expect: a 4-row table (bm25, dense, hybrid, hybrid_rerank), `wrote evals/results/<date>.json`,
and a small `unresolved questions` count. The first run downloads bge-reranker-base
(about 1.1 GB). The file name must NOT end in `-smoke`.

Look at `unresolved` in the JSON. Each id there has `sources`/`evidence` in
`evals/golden.jsonl` that matched no chunk. Fix the regex or source, or review the
question. Original-source questions stay unresolved until you add notes to
`data/raw/original/`.

## 3. Done check for the offline run (can fail)

```bash
uv run python scripts/check_results.py; echo "exit=$?"
```
Expect: `OK` and `exit=0`. `FAIL no non-smoke result` or `only N labeled questions`
means the corpus or index was not the real one.

## 4. Calibrate the refusal gate

Open `evals/results/<date>.json` and read `ablation.gate_sweep`: precision and recall
of refusal at each rerank-score threshold. Pick the threshold where recall is high and
precision stays high, then set it for later runs:

```bash
export MIN_RERANK_SCORE=<your pick>
```

## 5. Live baseline (spends money, hard cap $5)

Set the key in your shell only. Do not write it to `.env`.

```bash
export OPENROUTER_API_KEY=...   # your key
export GENERATION_MODEL=anthropic/claude-haiku-4.5
uv run python evals/run_evals.py --live --gen-configs dense,hybrid_rerank
```
Expect first: `projected live cost $X; remaining $5.00`. If X exceeds remaining, the run
aborts before any call. Lower the cost with `--gen-configs hybrid_rerank` or a
cheaper `--judge-model`. After the run: a table with Ragas columns filled, and
`live spend $...`. The ledger is `.traces/spend.json` (gitignored); it enforces the
cap across runs.

## 6. Done check for the live run (can fail)

```bash
uv run python scripts/check_results.py --live; echo "exit=$?"
```
Expect: `OK`, `exit=0`.

## 7. Record it

```bash
git add evals/results/<date>.json
```
Then copy the table into `ROADMAP.md` and `README.md` (`scripts/render_results.py`
does this from the JSON, see the README PR) and commit.
