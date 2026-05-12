# chef-rag

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/python-3.12+-blue.svg) ![Status](https://img.shields.io/badge/status-in%20development-orange)

A production-grade RAG system for professional culinary knowledge. Answers questions about cooking techniques, knife skills, kitchen management, food safety, and BOH operations from a curated expert corpus.

Built with hybrid retrieval (BM25 + dense vectors), cross-encoder reranking, a swappable model layer, and a full eval harness. Traced end-to-end with Langfuse.

---

## Why this domain

Most RAG demos use Wikipedia or PDF chatbots. This one is built on 10 years of professional kitchen knowledge that no CS applicant can replicate -- from line cook to Head Chef. The domain is the differentiator; the engineering is the argument.

---

## Architecture

```
User query
    │
    ▼
Hybrid Retrieval
    ├── BM25 (Postgres tsvector)        keyword precision
    └── Dense vectors (pgvector)        semantic recall
    │
    ▼
Cross-encoder reranker (bge-reranker-base)
    │
    ▼
Generation layer (OpenRouter -- model-swappable)
    │
    ▼
Response + citations
    │
    └── Langfuse trace (every request, cost, latency)
```

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `text-embedding-3-small` (OpenAI) or `nomic-embed-text` (local Ollama) | Cost vs. quality tradeoff, swappable |
| Vector store | Supabase pgvector | One DB for app data + vectors; free tier covers project |
| Keyword search | Postgres tsvector | No extra infra; hybrid retrieval in one query |
| Reranker | `bge-reranker-base` via HF Inference | Cheap, effective cross-encoder |
| LLM | OpenRouter (Claude/GPT/Llama swap) | Model-agnostic generation |
| Tracing | Langfuse (self-hosted) | Full observability: cost, latency, retrieval quality per request |
| Serving | Modal serverless | Pay-per-request, sub-second cold start |
| Frontend | Astro + chat island | Static site, citations as expandable cards |
| Runtime | Python 3.12, uv | Fast installs, lockfile-first |

---

## Corpus

**Public domain sources:**
- Project Gutenberg culinary classics (Escoffier, Fannie Farmer, Joy of Cooking era)
- USDA food safety and nutrition data (public domain)
- FDA food code documents
- NIH nutrition research (public access)
- ServSafe study guide (public portions)

**Original content:**
- Technique writeups from 10 years professional kitchen experience
- Kitchen management SOPs
- Mise en place frameworks
- BOH communication protocols

**Scope of v1 corpus:** ~200 documents, ~50K chunks at 500 tokens with 50-token overlap

---

## Eval harness

30 hand-curated golden Q&A pairs covering:
- Technique questions ("What's the Maillard reaction and at what temperature does it occur?")
- Safety questions ("What are the danger zone temperatures for food storage?")
- Management questions ("How do you run a pre-shift meeting for a line of 8?")
- Precision questions ("What is the internal temp for medium-rare beef?")

Metrics: faithfulness, answer relevancy, context precision (via Ragas)

**Baseline scores will be published here when v1 ships.**

---

## Project structure

```
chef-rag/
├── src/
│   ├── ingest.py          # Document loading, chunking, embedding
│   ├── retrieval.py       # Hybrid BM25 + dense retrieval
│   ├── rerank.py          # Cross-encoder reranking
│   ├── generate.py        # Generation with OpenRouter
│   └── trace.py           # Langfuse instrumentation
├── evals/
│   ├── golden.jsonl       # 30 curated Q&A pairs
│   └── run_evals.py       # Ragas eval runner
├── data/
│   ├── raw/               # Source documents
│   └── processed/         # Chunked, ready for embedding
├── docs/
│   └── design.md          # Extended design decisions
├── CLAUDE.md
├── pyproject.toml
└── README.md
```

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for what's built, what's next, and stretch targets.

---

## Running locally

```bash
# Install
uv sync

# Ingest documents
uv run python -m src.ingest --source data/raw/

# Query
uv run python -m src.cli query "how do I hold a chef knife correctly"

# Run evals
uv run python evals/run_evals.py
```

---

## Eval results

Tracked in [ROADMAP.md](./ROADMAP.md) and filled in as v1 ships.

---

_Solomon Smith -- built May 2026_
