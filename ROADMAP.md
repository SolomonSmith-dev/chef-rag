# chef-rag Roadmap

Tracking what's built, what's next, and stretch targets.

## Done

- [x] Design doc + repo scaffold
- [x] Foundation: `docs/design.md`, golden evals, `src/` package scaffold, CI
- [x] Ingestion: chunking (500/50, cl100k_base), stable chunk ids, fetch script and runbook
- [x] Hybrid retrieval (BM25 + dense, RRF k=60), local and Supabase backends, bge reranker
- [x] Citation-constrained generation, refusal gate, Langfuse or local JSONL tracing, spend cap
- [x] Eval harness: 68 golden questions, 4-way ablation, Ragas, refusal, cost

## In progress

- [ ] Real corpus download (run `docs/corpus-runbook.md`; sandbox could not reach the hosts)
- [ ] Review every `needs_review` golden question and add notes to `data/raw/original/`
- [ ] Live baseline (run `docs/eval-runbook.md`; needs corpus, model downloads and an OpenRouter key)

## Next

- [ ] Modal serverless deployment

## After v1

- [ ] Chat UI on Astro
- [ ] Eval iteration: improve scores by 10% from baseline
- [ ] Fine-tuned model variant

## Eval results (filled in as v1 ships)

| Version | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| v1 baseline | _not run_ | _not run_ | _not run_ |

Not run: the build sandbox had no network access to the corpus hosts, Hugging Face or OpenRouter, and no API keys. Spend so far: $0.00 of the $5 cap. Run `docs/eval-runbook.md` to produce `evals/results/<date>.json`.
