"""
Pydantic configuration loader and validator (src/core/config.py).

Provides strict runtime type checking, default values, environment variable
resolution, and clear startup validation errors for YAML configuration files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


# ==============================================================================
# 1. Sub-models matching config/app_config.yaml
# ==============================================================================

class LocalEmbeddingConfig(BaseModel):
    """Settings for running a free local HuggingFace SentenceTransformer model."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID used to generate embeddings locally.",
    )
    device: str = Field(
        default="cpu",
        description="Compute device: 'cpu', 'mps' (Apple Silicon), or 'cuda'.",
    )


class MistralEmbeddingConfig(BaseModel):
    """Settings for calling Mistral's cloud embedding API."""

    api_key: str = Field(
        default="",
        description="Optional inline API key; falls back to MISTRAL_API_KEY env var.",
    )
    model_name: str = Field(
        default="mistral-embed",
        description="Mistral embedding model name.",
    )
    timeout_sec: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)


class EmbeddingConfig(BaseModel):
    """
    Top-level embedding configuration.

    Acts as a Strategy selector: both "local" and "mistral" implement the same
    LangChain `Embeddings` interface, and `provider` selects the concrete backend
    instantiated at runtime.
    """

    provider: Literal["local", "mistral"] = "local"
    local: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)
    mistral: MistralEmbeddingConfig = Field(default_factory=MistralEmbeddingConfig)

    @model_validator(mode="after")
    def resolve_and_validate_mistral_key(self) -> "EmbeddingConfig":
        """
        Resolve MISTRAL_API_KEY from environment if YAML field is blank,
        and fail fast at startup if provider=='mistral' without a key.
        """
        env_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not self.mistral.api_key.strip() and env_key:
            self.mistral.api_key = env_key

        if self.provider == "mistral" and not self.mistral.api_key.strip():
            raise ValueError(
                "Configuration error: embedding.provider is set to 'mistral', "
                "but neither embedding.mistral.api_key in config/app_config.yaml "
                "nor MISTRAL_API_KEY in .env / environment is set."
            )
        return self


class DatabaseConfig(BaseModel):
    """
    Settings for our two-layer storage: SQL (PostgreSQL or SQLite) + Qdrant (vectors).
    """

    database_url: str = Field(
        default="",
        description=(
            "Optional full SQLAlchemy URL (e.g. postgresql+psycopg://...). "
            "Overridden by DATABASE_URL env var; falls back to `sqlite_path` when blank."
        ),
    )
    sqlite_path: str = Field(default="./data/wololo_state.db")
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_local_path: str = Field(default="./data/qdrant")
    qdrant_url: str = Field(default="http://localhost:6333")
    collection_prefix: str = Field(default="wololo_articles")

    @model_validator(mode="after")
    def apply_env_overrides(self) -> "DatabaseConfig":
        """
        Allow `.env` or Docker Compose environment variables to point the Indexer
        to external databases (`WololoRAG-Infra`) without changing YAML files.
        """
        if env_db_url := os.getenv("DATABASE_URL", "").strip():
            self.database_url = env_db_url
        if env_sqlite := os.getenv("SQLITE_PATH", "").strip():
            self.sqlite_path = env_sqlite
        if env_mode := os.getenv("QDRANT_MODE", "").strip():
            if env_mode in ("local", "server"):
                self.qdrant_mode = env_mode  # type: ignore[assignment]
        if env_url := os.getenv("QDRANT_URL", "").strip():
            self.qdrant_url = env_url
        if env_local_path := os.getenv("QDRANT_LOCAL_PATH", "").strip():
            self.qdrant_local_path = env_local_path
        return self

    @property
    def sqlalchemy_url(self) -> str:
        """Return explicit `database_url` if set, otherwise build a local SQLite URL."""
        if self.database_url.strip():
            return self.database_url.strip()
        return self.sqlite_path


class ScraperConfig(BaseModel):
    """Settings for fetching articles from Wololo.net WordPress REST API."""

    base_url: str = Field(default="https://wololo.net")
    request_timeout_sec: int = Field(default=20, gt=0)
    default_discover_batch: int = Field(default=20, gt=0, le=100)


class AppConfig(BaseModel):
    """Root schema for config/app_config.yaml."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)


# ==============================================================================
# 2. Sub-models matching config/vector_config.yaml
# ==============================================================================

class ChunkingConfig(BaseModel):
    """
    Controls whether articles are split into token-sized chunks before embedding.
    """

    enabled: bool = True
    chunk_size_tokens: int = Field(default=400, gt=0, le=8192)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    tokenizer_encoding: str = Field(default="cl100k_base")

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        """Overlap must be strictly smaller than chunk size to avoid infinite loops."""
        if self.enabled and self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"Invalid chunking config: chunk_overlap_tokens ({self.chunk_overlap_tokens}) "
                f"must be smaller than chunk_size_tokens ({self.chunk_size_tokens})."
            )
        return self


class SearchConfig(BaseModel):
    """Controls how Qdrant searches, deduplicates, and formats vector matches."""

    top_k: int = Field(default=5, gt=0, le=50)
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    distance_metric: Literal["Cosine", "Dot", "Euclid"] = "Cosine"
    max_chunks_per_article: int = Field(default=1, ge=1, le=10)
    hybrid_keyword_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    context_mode: Literal["best_chunk", "full_article"] = "best_chunk"


class VectorConfig(BaseModel):
    """Root schema for config/vector_config.yaml."""

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)


# ==============================================================================
# 3. Unified Settings Loader
# ==============================================================================

class FullSettings(BaseModel):
    """Combined validated settings object passed across the application."""

    app: AppConfig
    vector: VectorConfig


def _read_yaml(path: Path) -> dict:
    """Safely load a YAML file into a Python dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} must be a key-value mapping.")
    return data


def load_settings(
    app_config_path: str | Path | None = None,
    vector_config_path: str | Path | None = None,
    dotenv_path: str | Path | None = None,
) -> FullSettings:
    """
    Load .env variables and validate both YAML config files via Pydantic.

    Step 1: Load environment variables from `.env` (if present).
    Step 2: Resolve paths for `app_config.yaml` and `vector_config.yaml`.
    Step 3: Parse YAML and validate against Pydantic models.
    """
    load_dotenv(dotenv_path=dotenv_path)

    project_root = Path(__file__).resolve().parents[2]
    resolved_app_path = Path(
        app_config_path or os.getenv("APP_CONFIG_PATH") or (project_root / "config" / "app_config.yaml")
    )
    resolved_vector_path = Path(
        vector_config_path or os.getenv("VECTOR_CONFIG_PATH") or (project_root / "config" / "vector_config.yaml")
    )

    app_raw = _read_yaml(resolved_app_path)
    vector_raw = _read_yaml(resolved_vector_path)

    return FullSettings(
        app=AppConfig.model_validate(app_raw),
        vector=VectorConfig.model_validate(vector_raw),
    )
