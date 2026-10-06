# chef-rag Roadmap

Tracking what's built, what's next, and stretch targets.

## Done

- [x] Design doc + repo scaffold
- [x] Foundation: `docs/design.md`, golden evals, `src/` package scaffold, CI
- [x] Ingestion: chunking (500/50, cl100k_base), stable chunk ids, fetch script and runbook
- [x] Hybrid retrieval (BM25 + dense, RRF k=60), local and Supabase backends, bge reranker
- [x] Citation-constrained generation, refusal gate, Langfuse or local JSONL tracing, spend cap
- [x] Eval harness: 68 golden questions, 4-way ablation, Ragas, refusal, cost
- [x] Demo service (FastAPI, rate limit, daily token cap, no question logging) and HF Space deploy kit
- [x] README with CI and eval badges; results block rendered from committed JSON

## In progress

- [ ] Deploy the Hugging Face Space (`docs/deploy-hf.md`) and replace the README demo placeholder
- [ ] Real corpus download (run `docs/corpus-runbook.md`; sandbox could not reach the hosts)
- [ ] Review every `needs_review` golden question and add notes to `data/raw/original/`
- [ ] Live baseline (run `docs/eval-runbook.md`; needs corpus, model downloads and an OpenRouter key)

## Next

- [ ] Review golden labels after the first real run; fix `unresolved` questions

## After v1

- [ ] Chat UI on Astro
- [ ] Eval iteration: improve scores by 10% from baseline
- [ ] Fine-tuned model variant

## Eval results (filled in as v1 ships)

<!-- baseline:start -->
| Version | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| v1 baseline | _not run_ | _not run_ | _not run_ |
<!-- baseline:end -->

Not run: the build sandbox had no network access to the corpus hosts, Hugging Face or OpenRouter, and no API keys. Spend so far: $0.00 of the $5 cap. Run `docs/eval-runbook.md` to produce `evals/results/<date>.json`.
