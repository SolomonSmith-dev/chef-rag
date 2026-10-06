# chef-rag Design

Extended design decisions for v1. README covers the product narrative; this doc covers schemas, contracts, and module boundaries.

## Chunking

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target size | 500 tokens | Fits reranker context; one technique per chunk |
| Overlap | 50 tokens | Preserves boundary context across splits |
| Splitter | Recursive character | LangChain-compatible; handles markdown and plain text |
| Tokenizer | `cl100k_base` via tiktoken | Matches embedding model tokenization |

Ingest writes chunks to `data/processed/` as JSONL before upsert to Supabase.

## Supabase schema

Enable extensions:

```sql
create extension if not exists vector;
```

### `documents`

Source-level metadata. One row per ingested file.

```sql
create table documents (
  id uuid primary key default gen_random_uuid(),
  source_path text not null unique,
  title text,
  source_type text not null,  -- gutenberg | fda | usda | original | other
  ingested_at timestamptz not null default now()
);
```

### `chunks`

Retrieval unit. Hybrid search uses `content_tsv` (BM25) and `embedding` (dense).

```sql
create table chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  chunk_index int not null,
  content text not null,
  content_tsv tsvector generated always as (to_tsvector('english', content)) stored,
  embedding vector(1536),  -- text-embedding-3-small; adjust if model changes
  token_count int not null,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index chunks_embedding_idx on chunks
  using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create index chunks_content_tsv_idx on chunks using gin (content_tsv);
```

### Embedding dimension and backends

The schema above uses `vector(1536)` for `text-embedding-3-small`. The default local
embedder (`sentence-transformers/all-MiniLM-L6-v2`) outputs **384** dimensions, so
dimension is a setting (`EMBEDDING_DIM`, default 384), not a constant.
`supabase/migrations/` is rendered from `src/schema.sql.tmpl` by
`src.retrieval.render_migration(dim)`; re-render when the model changes. A local index
records its dimension and refuses to load with a mismatched embedder.

Deviations from the sketch above: `chunks` gains a `chunk_id text unique` column (the
citation id), denormalized `source_path`/`title`/`source_type` (so RPC results need no
join), and a nullable `document_id`. Dense and BM25 candidate queries are the SQL
functions `match_chunks_dense` and `match_chunks_bm25`.

`src/retrieval.py` exposes two backends behind one interface. `local` (default for CI,
demo, offline) uses rank-bm25 and a numpy matrix. `supabase` uses the RPCs above.
Both fuse with the same RRF code. `--mode bm25|dense` exists for ablations only.

### Hybrid retrieval (v1)

RRF fusion of BM25 and dense ranks in application code (`src/retrieval.py`):

1. Dense: `order by embedding <=> $query_embedding limit 20`
2. BM25: `order by ts_rank(content_tsv, plainto_tsquery('english', $query)) desc limit 20`
3. Fuse with reciprocal rank fusion (k=60)
4. Return top 10 to reranker

Never dense-only. BM25-only is acceptable for debugging only.

## Module boundaries

| Module | Responsibility |
|--------|----------------|
| `src/config.py` | Env loading, validated settings |
| `src/ingest.py` | Load raw docs, chunk, embed, upsert |
| `src/retrieval.py` | Hybrid search against Supabase |
| `src/rerank.py` | Cross-encoder reranking (`bge-reranker-base`) |
| `src/generate.py` | OpenRouter completion with citation prompt |
| `src/trace.py` | Langfuse span wrappers; every LLM call traced |
| `src/cli.py` | `ingest` and `query` subcommands |

## Generation contract

Prompt structure:

1. System: answer only from provided context; cite every claim as `[cite: <chunk_id>]`;
   reply exactly `NOT_IN_SOURCES` when the passages do not answer; when a `gutenberg`
   source conflicts with `fda`/`usda` on safety, temperature or time, follow `fda`/`usda`
   and say the older text differs.
2. User: question + top-k reranked chunks, each labeled with chunk_id and source_type.
3. Output: answer text with inline `[cite: ...]` markers.

Enforcement in `src/generate.py`:

- A citation not in the retrieved set raises `InvalidCitationError` (hard error).
- An answer with no citation raises `MissingCitationError`.
- Refusal ("Not in my sources") when retrieval is empty, when the best rerank score is
  below `MIN_RERANK_SCORE` (checked before any LLM call), or when the model returns
  `NOT_IN_SOURCES`. The default threshold 0.0 is uncalibrated; the eval JSON includes a
  `gate_sweep` of refusal precision/recall per threshold to choose it.

All LLM calls go through OpenRouter (`openai` client with custom base URL) and through
`SpendLedger` (`src/budget.py`), which enforces the USD cap before each call.

## Tracing

`src/trace.py` `make_tracer` picks Langfuse when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set, otherwise a local JSONL tracer writing to `.traces/`
(gitignored, override with `TRACE_DIR`). The CLI and generation path always go through
a tracer: spans `query`, `retrieve`, `rerank`, `answer`, `generate` (kind generation,
with model, token usage and cost). `redact_input=True` drops inputs and outputs (used by
the public demo). Ragas judge calls run under the eval harness, are recorded in the
spend ledger via a callback, and use estimated (not reported) cost.

## Eval contract

Golden pairs live in `evals/golden.jsonl`. Each line:

```json
{
  "id": "safety-004",
  "category": "safety",
  "question": "...",
  "reference_answer": "...",
  "tags": ["cold-holding", "fda"],
  "answerable": true,
  "sources": ["fda"],
  "evidence": ["\\b41\\s*\u00b0?\\s*F\\b"],
  "needs_review": true
}
```

Categories: `technique`, `safety`, `management`, `precision` (ratios and temperatures),
`outdated` (old text conflicts with modern guidance; the right answer prefers fda/usda),
`out_of_corpus` (`answerable: false`; the right behavior is refusal).

Relevant chunks are resolved at run time: a chunk is relevant when its `source_path`
starts with one of `sources` (or `original` matches `source_type`) and its text matches
an `evidence` regex. Questions with no matching chunk are reported as `unresolved` and
excluded from retrieval metrics. `needs_review: true` marks items drafted by tooling;
`labels_need_review` marks the author's original items whose labels were drafted.

`evals/run_evals.py` reports:

- Retrieval, for bm25 / dense / hybrid / hybrid+rerank: recall@10 (before rerank),
  recall@k (after), MRR. Recall@N = |relevant in top N| / min(|relevant|, N).
- Live (`--live`): Ragas faithfulness, answer relevancy, context precision; refusal
  precision and recall; citation validity rate; modern-source rate on `outdated`
  questions; p50/p95 latency; cost per query; spend.
- A run with the hashing embedder or no reranker is marked `smoke: true` and written to
  `<date>-smoke.json`; it is never a quality result.

Baseline scores are recorded in `ROADMAP.md` from `evals/results/<date>.json`.

## Environment

Required vars (see `.env.example`):

- `OPENROUTER_API_KEY`
- `SUPABASE_URL`, `SUPABASE_KEY`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

Optional:

- `EMBEDDING_MODEL` (default: `openai/text-embedding-3-small` via OpenRouter)
- `GENERATION_MODEL` (default: `anthropic/claude-sonnet-4`)

## v1 non-goals

- Modal deployment (post-CLI)
- Astro frontend (post-v1)
- Fine-tuned models
- Dense-only retrieval
