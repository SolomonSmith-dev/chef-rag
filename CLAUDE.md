# Project: chef-rag

- Lang/runtime: Python 3.12, uv
- Test: `uv run pytest -q`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Type check: `uv run mypy src/`
- Run: `uv run python -m src.cli`
- Style: black-compatible, type hints required, pytest
- Naming: snake_case files/functions, PascalCase classes
- Never: commit secrets, write .env, run destructive shell without dry-run
- Always: TDD red-green-refactor
- Branch policy: feature branches off main, PR required, squash merge
- Dependencies: add via `uv add`, pin in pyproject.toml

## Key paths
- Golden evals: evals/golden.jsonl
- Raw corpus: data/raw/
- Processed chunks: data/processed/

## Env vars required
- OPENROUTER_API_KEY
- SUPABASE_URL
- SUPABASE_KEY
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY

## Architecture notes
- Retrieval is hybrid: BM25 (tsvector) + dense (pgvector). Never use dense-only.
- All LLM calls go through OpenRouter for model-swappability.
- Every request MUST emit a Langfuse trace. No dark calls.
- Chunking: 500 tokens, 50 overlap, recursive character splitter.
