FROM python:3.12-slim

RUN pip install --no-cache-dir uv \
 && useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock ./
COPY --chown=user src ./src
COPY --chown=user scripts ./scripts
COPY --chown=user data ./data
RUN uv sync --frozen

# Build the corpus and the local index into the image. Fails the build if any source
# is missing or wrong, so a broken deploy never starts. First build downloads the
# MiniLM and bge-reranker weights into the image cache.
RUN uv run python scripts/fetch_corpus.py \
 && uv run python scripts/fetch_corpus.py --verify \
 && uv run python -m src.cli ingest --source data/raw/ --out data/processed/ --index \
 && uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

# OPENROUTER_API_KEY must be a Space secret. Optional: LANGFUSE_PUBLIC_KEY,
# LANGFUSE_SECRET_KEY, LANGFUSE_HOST, DEMO_MODEL, DEMO_DAILY_TOKEN_CAP,
# DEMO_RATE_PER_MIN, DEMO_RATE_PER_DAY.
ENV INDEX_DIR=data/processed/index CHUNKS_PATH=data/processed/chunks.jsonl
EXPOSE 7860
CMD ["uv", "run", "uvicorn", "--factory", "src.demo:build_app", "--host", "0.0.0.0", "--port", "7860"]
