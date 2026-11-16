"""
Typed state schemas for LangGraph workflows (src/core/graphs/state.py).

Each node function receives the current `State` dictionary and returns only
the fields it updates.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document

from src.core.db.models import ArticleRecord


class ArticleIngestionState(TypedDict, total=False):
    """Shared state passed between nodes in the Article Vectorization Graph."""

    # Input: SQLAlchemy row fetched from SQLite/PostgreSQL (Phase 1 -> Phase 2 handoff)
    article: ArticleRecord
    active_provider: str

    # Computed intermediate values
    content_hash: str
    chunking_enabled: bool
    chunks: list[Document]
    vectors: list[list[float]]
    vector_ids: list[str]

    # Output / error tracking (saved back to SQL)
    status: str
    error_str: str | None


class RetrievalState(TypedDict, total=False):
    """Shared state passed between nodes in the RAG Search / Retrieval Graph."""

    # Inputs
    query: str
    topic_filter: str | None
    top_k: int | None
    published_after: str | None

    # Step 1: Query converted into float[] coordinates
    query_vector: list[float]

    # Step 2: Raw nearest-neighbor hits from Qdrant
    raw_hits: list[dict[str, Any]]

    # Step 3: Hits enriched with authoritative metadata & status from SQLite
    enriched_hits: list[dict[str, Any]]

    # Step 4: Clean context block formatted for LLM / MCP / CLI output
    formatted_context: str
