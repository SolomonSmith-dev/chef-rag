-- chef-rag schema. Rendered from src/schema.sql.tmpl by src.retrieval.render_migration.
-- Embedding dimension: 384 (all-MiniLM-L6-v2). docs/design.md used 1536 for
-- text-embedding-3-small; re-render with another dimension if the model changes.
create extension if not exists vector;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  source_path text not null unique,
  title text,
  source_type text not null,  -- gutenberg | fda | usda | original | other
  ingested_at timestamptz not null default now()
);

create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  chunk_id text not null unique,
  document_id uuid references documents(id) on delete cascade,
  source_path text not null,
  title text,
  source_type text not null default 'other',
  chunk_index int not null,
  content text not null,
  content_tsv tsvector generated always as (to_tsvector('english', content)) stored,
  embedding vector(384),
  token_count int not null,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists chunks_embedding_idx on chunks
  using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create index if not exists chunks_content_tsv_idx on chunks using gin (content_tsv);

-- Dense candidates: order by embedding <=> query_embedding.
create or replace function match_chunks_dense(
  query_embedding vector(384),
  match_count int default 20
) returns table (
  chunk_id text, content text, source_path text, title text, source_type text, score float
) language sql stable as $$
  select c.chunk_id, c.content, c.source_path, c.title, c.source_type,
         1 - (c.embedding <=> query_embedding) as score
  from chunks c
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- BM25-style candidates: ts_rank over the generated tsvector.
create or replace function match_chunks_bm25(
  query_text text,
  match_count int default 20
) returns table (
  chunk_id text, content text, source_path text, title text, source_type text, score float
) language sql stable as $$
  select c.chunk_id, c.content, c.source_path, c.title, c.source_type,
         ts_rank(c.content_tsv, plainto_tsquery('english', query_text))::float as score
  from chunks c
  where c.content_tsv @@ plainto_tsquery('english', query_text)
  order by score desc
  limit match_count;
$$;
