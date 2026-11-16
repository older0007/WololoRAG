"""
Repository layer over SQLite / PostgreSQL via SQLAlchemy (src/core/db/repository.py).

Encapsulates SQL queries and state transitions behind a clean, typed interface.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from src.core.db.models import ArticleRecord, ArticleStatus, Base, utc_now


def compute_content_hash(text: str | None) -> str:
    """Compute a deterministic SHA-256 hash for article text change detection."""
    normalized = (text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _to_naive_utc(dt: datetime | None) -> datetime | None:
    """
    Normalize datetimes to naive UTC for safe comparison across SQLite reads.
    SQLite does not store timezone offsets natively, so returning rows often
    have tzinfo=None while API responses have tzinfo=timezone.utc.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class DatabaseStats(BaseModel):
    """Snapshot of SQLite article synchronization metrics shown on startup."""

    total_in_sql: int = 0
    discovered_count: int = 0
    indexed_count: int = 0
    indexed_for_active_provider: int = 0
    indexed_other_provider: int = 0
    outdated_count: int = 0
    failed_count: int = 0

    @property
    def pending_for_vectorization(self) -> int:
        """Total articles waiting to be embedded for the current active provider."""
        return self.discovered_count + self.outdated_count + self.indexed_other_provider


class ArticleRepository:
    """Thread-safe CRUD and reporting interface for the `articles` SQL table (SQLite or PostgreSQL)."""

    def __init__(self, db_target: str | Path) -> None:
        """
        Step 1: Accept either a full SQLAlchemy URL (`postgresql+psycopg://...` / `sqlite:///...`)
                or a local SQLite filesystem path (`./data/wololo_state.db`).
        Step 2: Configure SQLAlchemy engine (only SQLite needs `check_same_thread=False`).
        Step 3: Auto-create tables on first launch.
        """
        target_str = str(db_target).strip()
        if "://" in target_str:
            self.db_url = target_str
            if self.db_url.startswith("sqlite:///") and not self.db_url.startswith("sqlite:///:memory:"):
                raw_path = Path(self.db_url.removeprefix("sqlite:///"))
                raw_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            db_path = Path(target_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_url = f"sqlite:///{db_path}"

        is_sqlite = self.db_url.startswith("sqlite:")
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        self.engine = create_engine(
            self.db_url,
            connect_args=connect_args,
            echo=False,
        )
        # expire_on_commit=False lets callers read fields after the session closes.
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @property
    def backend_name(self) -> str:
        """Human-readable SQL backend label for startup diagnostics."""
        if self.db_url.startswith("postgresql"):
            return "PostgreSQL"
        return "SQLite"

    def _session(self) -> Session:
        return self._session_factory()

    def get_stats(self, active_provider: str) -> DatabaseStats:
        """
        Aggregate row counts by status and active embedding provider.
        Used at startup to tell the user what is already indexed vs pending.
        """
        with self._session() as session:
            rows = session.execute(
                select(ArticleRecord.status, ArticleRecord.embedding_provider, func.count())
                .group_by(ArticleRecord.status, ArticleRecord.embedding_provider)
            ).all()

        stats = DatabaseStats()
        for status_val, provider_val, count in rows:
            stats.total_in_sql += count
            if status_val == ArticleStatus.DISCOVERED.value:
                stats.discovered_count += count
            elif status_val == ArticleStatus.INDEXED.value:
                stats.indexed_count += count
                if provider_val == active_provider:
                    stats.indexed_for_active_provider += count
                else:
                    stats.indexed_other_provider += count
            elif status_val == ArticleStatus.OUTDATED.value:
                stats.outdated_count += count
            elif status_val == ArticleStatus.FAILED.value:
                stats.failed_count += count

        return stats

    def upsert_discovered_articles(self, items: Iterable[dict]) -> dict[str, int]:
        """
        Phase 1 Ingestion: Insert newly fetched Wololo.net articles as DISCOVERED,
        or mark existing INDEXED articles as OUTDATED if remote content changed.

        Returns a summary dictionary: {"inserted": N, "updated_to_outdated": M, "unchanged": K}.
        """
        summary = {"inserted": 0, "updated_to_outdated": 0, "unchanged": 0}

        with self._session() as session:
            for item in items:
                wp_post_id = int(item["wp_post_id"])
                cleaned_content = item.get("cleaned_content") or ""
                new_hash = compute_content_hash(cleaned_content)
                remote_mod = item.get("remote_modified_at")

                existing = session.get(ArticleRecord, wp_post_id)
                if existing is None:
                    record = ArticleRecord(
                        wp_post_id=wp_post_id,
                        title=item["title"],
                        url=item["url"],
                        topic=item.get("topic") or "General",
                        published_at=item.get("published_at"),
                        remote_modified_at=remote_mod,
                        cleaned_content=cleaned_content,
                        content_hash=new_hash,
                        status=ArticleStatus.DISCOVERED.value,
                        vector_ids=[],
                        chunk_count=0,
                        error_str=None,
                    )
                    session.add(record)
                    summary["inserted"] += 1
                else:
                    content_changed = existing.content_hash != new_hash
                    remote_newer = (
                        _to_naive_utc(remote_mod) is not None
                        and _to_naive_utc(existing.remote_modified_at) is not None
                        and _to_naive_utc(remote_mod) > _to_naive_utc(existing.remote_modified_at)  # type: ignore[operator]
                    )

                    existing.title = item["title"]
                    existing.url = item["url"]
                    existing.topic = item.get("topic") or existing.topic
                    existing.published_at = item.get("published_at") or existing.published_at
                    existing.remote_modified_at = remote_mod or existing.remote_modified_at
                    existing.cleaned_content = cleaned_content
                    existing.content_hash = new_hash

                    if existing.status == ArticleStatus.INDEXED.value and (content_changed or remote_newer):
                        existing.status = ArticleStatus.OUTDATED.value
                        summary["updated_to_outdated"] += 1
                    else:
                        summary["unchanged"] += 1

            session.commit()

        return summary

    def get_pending_for_vectorization(
        self,
        limit: int,
        active_provider: str,
        include_failed: bool = False,
    ) -> list[ArticleRecord]:
        """
        Phase 2 Selection: Fetch up to `limit` articles ready to be vectorized.

        Priority order:
        1. OUTDATED articles (already indexed before, need refresh)
        2. DISCOVERED articles (freshly synced from Wololo, never vectorized)
        3. INDEXED articles created by a *different* embedding provider
           (e.g., previously embedded locally, now user switched to Mistral API)
        4. FAILED articles (only when `include_failed=True` for manual retries)
        """
        if limit <= 0:
            return []

        selected: list[ArticleRecord] = []
        seen_ids: set[int] = set()

        def _append_batch(records: Iterable[ArticleRecord]) -> None:
            for rec in records:
                if len(selected) >= limit:
                    break
                if rec.wp_post_id not in seen_ids:
                    seen_ids.add(rec.wp_post_id)
                    selected.append(rec)

        with self._session() as session:
            # Priority 1 & 2: OUTDATED then DISCOVERED (newest published first)
            for target_status in (ArticleStatus.OUTDATED.value, ArticleStatus.DISCOVERED.value):
                if len(selected) >= limit:
                    break
                stmt = (
                    select(ArticleRecord)
                    .where(ArticleRecord.status == target_status)
                    .order_by(ArticleRecord.published_at.desc(), ArticleRecord.wp_post_id.desc())
                    .limit(limit - len(selected))
                )
                _append_batch(session.scalars(stmt).all())

            # Priority 3: Indexed under a different embedding provider
            if len(selected) < limit:
                stmt = (
                    select(ArticleRecord)
                    .where(
                        ArticleRecord.status == ArticleStatus.INDEXED.value,
                        ArticleRecord.embedding_provider != active_provider,
                    )
                    .order_by(ArticleRecord.published_at.desc(), ArticleRecord.wp_post_id.desc())
                    .limit(limit - len(selected))
                )
                _append_batch(session.scalars(stmt).all())

            # Priority 4: Optional retry of FAILED rows
            if include_failed and len(selected) < limit:
                stmt = (
                    select(ArticleRecord)
                    .where(ArticleRecord.status == ArticleStatus.FAILED.value)
                    .order_by(ArticleRecord.updated_at.asc())
                    .limit(limit - len(selected))
                )
                _append_batch(session.scalars(stmt).all())

        return selected

    def get_failed_articles(self, limit: int = 20) -> list[ArticleRecord]:
        """Return articles currently in FAILED state along with their `error_str`."""
        with self._session() as session:
            stmt = (
                select(ArticleRecord)
                .where(ArticleRecord.status == ArticleStatus.FAILED.value)
                .order_by(ArticleRecord.updated_at.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())

    def mark_article_indexed(
        self,
        wp_post_id: int,
        vector_ids: list[str],
        chunk_count: int,
        provider: str,
        content_hash: str,
    ) -> None:
        """
        Mark an article as INDEXED after its vectors are safely stored in Qdrant.
        Clears any previous `error_str` so the record is clean.
        """
        with self._session() as session:
            record = session.get(ArticleRecord, wp_post_id)
            if record is None:
                raise KeyError(f"Article wp_post_id={wp_post_id} not found in SQLite.")

            record.status = ArticleStatus.INDEXED.value
            record.vector_ids = vector_ids
            record.chunk_count = chunk_count
            record.embedding_provider = provider
            record.content_hash = content_hash
            record.error_str = None
            record.last_indexed_at = utc_now()
            session.commit()

    def mark_article_failed(self, wp_post_id: int, error_message: str) -> None:
        """
        Record a processing failure in SQLite without crashing the rest of the batch.
        Stores the exact exception details in `error_str` for debugging and retry.
        """
        with self._session() as session:
            record = session.get(ArticleRecord, wp_post_id)
            if record is None:
                return
            record.status = ArticleStatus.FAILED.value
            record.error_str = error_message[:4000]
            session.commit()

    def get_by_ids(self, wp_post_ids: Iterable[int]) -> dict[int, ArticleRecord]:
        """Hydrate multiple articles by ID in a single SQL IN query."""
        ids = list(set(wp_post_ids))
        if not ids:
            return {}
        with self._session() as session:
            stmt = select(ArticleRecord).where(ArticleRecord.wp_post_id.in_(ids))
            rows = session.scalars(stmt).all()
            return {row.wp_post_id: row for row in rows}

    def get_existing_wp_ids(self) -> set[int]:
        """Return the set of all WordPress post IDs currently tracked in SQLite."""
        with self._session() as session:
            rows = session.scalars(select(ArticleRecord.wp_post_id)).all()
            return set(rows)

    def clear_all_articles(self) -> int:
        """Delete all rows from the `articles` SQL table and return removed count."""
        with self._session() as session:
            result = session.execute(delete(ArticleRecord))
            session.commit()
            return int(result.rowcount or 0)
