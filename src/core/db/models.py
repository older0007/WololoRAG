"""
SQLAlchemy ORM models for the relational state database (src/core/db/models.py).

While Qdrant (Vector DB) stores mathematical vectors and chunk payloads, the SQL
database (`articles` table) acts as the authoritative registry for every Wololo.net
article:
- Which topic/category it belongs to
- Current lifecycle status (DISCOVERED, INDEXED, OUTDATED, FAILED)
- Exact list of Qdrant point UUIDs (`vector_ids`) owned by the article
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class ArticleStatus(str, Enum):
    """
    Lifecycle states for a Wololo.net article in our Two-Phase Ingestion pipeline.

    Phase 1 (Catalog Sync):
        DISCOVERED -> Metadata & cleaned text saved to SQLite; waiting to be vectorized.
    Phase 2 (LangGraph Vectorization):
        INDEXED    -> Chunks embedded and stored in Qdrant; `vector_ids` recorded.
        OUTDATED   -> Article changed on Wololo.net after last indexing; needs refresh.
        FAILED     -> Processing failed; exact details saved in `error_str`.
    """

    DISCOVERED = "DISCOVERED"
    INDEXED = "INDEXED"
    OUTDATED = "OUTDATED"
    FAILED = "FAILED"


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


class ArticleRecord(Base):
    """
    Relational row representing a single Wololo.net article / topic.
    """

    __tablename__ = "articles"

    # Original WordPress Post ID from Wololo.net (natural primary key).
    wp_post_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)

    # Human-readable article metadata.
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    topic: Mapped[str] = mapped_column(String(128), nullable=False, default="General", index=True)

    # Timestamps from WordPress API used to detect remote updates.
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    remote_modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Cleaned plain-text body extracted from WordPress HTML content.
    cleaned_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    # SHA-256 fingerprint of cleaned_content to detect text changes reliably.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Synchronization state between SQL DB and Vector DB.
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ArticleStatus.DISCOVERED.value,
        index=True,
    )

    # Exact list of Qdrant point UUIDs generated for this article's chunks.
    # Stored as JSON array so we can delete/replace stale vectors cleanly on updates.
    vector_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Tracks which embedding provider ("local" or "mistral") generated `vector_ids`.
    embedding_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Detailed error message if any node in the ingestion graph failed; NULL on success.
    error_str: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Audit timestamps.
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def __repr__(self) -> str:
        return (
            f"<ArticleRecord wp_post_id={self.wp_post_id} "
            f"topic={self.topic!r} status={self.status!r} chunks={self.chunk_count}>"
        )
