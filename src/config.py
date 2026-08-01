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
