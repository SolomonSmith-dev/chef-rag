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
  source_type text not null,  -- gutenberg | usda | original | other
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

1. System: answer only from provided context; cite chunk IDs
2. User: question + top-k reranked chunks
3. Output: answer text + list of `chunk_id` citations

All LLM calls go through OpenRouter (`openai` client with custom base URL).

## Eval contract

Golden pairs live in `evals/golden.jsonl`. Each line:

```json
{
  "id": "safety-001",
  "category": "safety",
  "question": "...",
  "reference_answer": "...",
  "tags": ["temperature", "fda"]
}
```

`evals/run_evals.py` runs Ragas metrics (faithfulness, answer relevancy, context precision) against the live pipeline. Baseline scores recorded in `ROADMAP.md`.

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
