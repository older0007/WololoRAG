"""
Embedding provider factory (src/core/vector/embeddings.py).

Implements a clean Strategy factory over LangChain's `Embeddings` interface
(`embed_documents` and `embed_query`), keeping the rest of the pipeline
decoupled from whether Local HuggingFace or Mistral API is active.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.core.config import AppConfig


def get_collection_name(app_config: AppConfig) -> str:
    """
    Build a provider-specific Qdrant collection name.

    Why this matters:
    - Local MiniLM produces 384-dimensional vectors (float[384]).
    - Mistral `mistral-embed` produces 1024-dimensional vectors (float[1024]).
    Separating collections (`wololo_articles_local` vs `wololo_articles_mistral`)
    lets you switch providers in config/app_config.yaml safely without collisions!
    """
    prefix = app_config.database.collection_prefix.strip()
    provider = app_config.embedding.provider.strip()
    return f"{prefix}_{provider}"


def create_embedding_model(app_config: AppConfig) -> Embeddings:
    """
    Instantiate the concrete LangChain `Embeddings` adapter configured in YAML.

    Imports are lazy-loaded inside each branch so local runs don't fail if
    Mistral isn't configured, and vice versa.
    """
    provider = app_config.embedding.provider

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        local_cfg = app_config.embedding.local
        return HuggingFaceEmbeddings(
            model_name=local_cfg.model_name,
            model_kwargs={"device": local_cfg.device},
            # Normalizing embeddings ensures Cosine Similarity == Dot Product.
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        from pydantic import SecretStr

        mistral_cfg = app_config.embedding.mistral
        return MistralAIEmbeddings(
            model=mistral_cfg.model_name,
            api_key=SecretStr(mistral_cfg.api_key),
            max_retries=mistral_cfg.max_retries,
            timeout=mistral_cfg.timeout_sec,
        )

    raise ValueError(f"Unsupported embedding provider: {provider!r}")


def probe_embedding_dimension(embeddings: Embeddings) -> int:
    """
    Determine the vector dimensionality (e.g. 384 or 1024) dynamically by
    running a single short probe query. Only called when initializing a brand-new
    Qdrant collection.
    """
    probe_vector = embeddings.embed_query("dimension probe")
    return len(probe_vector)
