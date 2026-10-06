"""Application configuration loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings for chef-rag services."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = Field(min_length=1)
    supabase_url: str = Field(min_length=1)
    supabase_key: str = Field(min_length=1)
    langfuse_public_key: str = Field(min_length=1)
    langfuse_secret_key: str = Field(min_length=1)
    langfuse_host: str = "http://localhost:3000"

    embedding_model: str = "openai/text-embedding-3-small"
    generation_model: str = "anthropic/claude-sonnet-4"

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50


def load_settings() -> Settings:
    """Load settings from the environment."""
    return Settings()


class RetrievalSettings(BaseSettings):
    """Retrieval knobs. No secrets, so the local backend runs without any API keys."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    backend: str = Field(default="local", validation_alias="RETRIEVAL_BACKEND")
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Must match the model above and the vector(N) column in supabase/migrations.
    # all-MiniLM-L6-v2 is 384; docs/design.md used 1536 for text-embedding-3-small.
    embedding_dim: int = 384
    rerank_model: str = "BAAI/bge-reranker-base"
    index_dir: str = "data/processed/index"
    chunks_path: str = "data/processed/chunks.jsonl"
    fuse_top_n: int = 10
    final_top_k: int = 5
    # Refuse when the best rerank score is below this. bge-reranker-base emits raw logits;
    # 0.0 is the sigmoid-0.5 point and is NOT calibrated: use gate_sweep in the eval JSON.
    min_rerank_score: float = 0.0
    generation_model: str = "anthropic/claude-sonnet-4"
    trace_dir: str = ".traces"
    spend_ledger: str = ".traces/spend.json"
    spend_cap_usd: float = 5.0


def load_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()
