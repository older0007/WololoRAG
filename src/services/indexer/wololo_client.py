"""
Wololo.net WordPress REST API client and HTML cleaner (src/services/indexer/wololo_client.py).

Why WordPress REST API instead of raw HTML scraping?
    Wololo.net runs on WordPress, which exposes structured JSON at:
    - `/wp-json/wp/v2/posts`      (articles, modified dates, categories, rendered HTML)
    - `/wp-json/wp/v2/categories` (human-readable topic names like PS5, PS4, Switch)
    This avoids brittle CSS selector scraping for pagination and gives exact post IDs!
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.core.config import ScraperConfig


class WololoClient:
    """HTTP client for fetching and cleaning articles from Wololo.net."""

    def __init__(self, config: ScraperConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.timeout_sec = config.request_timeout_sec
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "WololoRAG-Educational/0.1 (+https://github.com/older0007/WololoRAG)",
                "Accept": "application/json",
            }
        )
        self._category_cache: dict[int, str] | None = None

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_remote_total_posts(self) -> int:
        """
        Fetch 1 post header to read `X-WP-Total` — the total number of published
        articles currently hosted on Wololo.net.
        """
        resp = self.session.get(
            self._endpoint("/wp-json/wp/v2/posts"),
            params={"per_page": 1, "_fields": "id"},
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        total_header = resp.headers.get("X-WP-Total")
        if total_header is None:
            raise RuntimeError("Wololo.net API response did not include X-WP-Total header.")
        return int(total_header)

    def fetch_categories(self, force_refresh: bool = False) -> dict[int, str]:
        """
        Fetch and cache WordPress category ID -> Topic Name mappings.
        """
        if self._category_cache is not None and not force_refresh:
            return self._category_cache

        mapping: dict[int, str] = {}
        page = 1
        while True:
            resp = self.session.get(
                self._endpoint("/wp-json/wp/v2/categories"),
                params={"per_page": 100, "page": page, "_fields": "id,name"},
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            items = resp.json()
            if not items:
                break
            for cat in items:
                mapping[int(cat["id"])] = html.unescape(str(cat["name"])).strip()

            total_pages = int(resp.headers.get("X-WP-TotalPages", "1"))
            if page >= total_pages:
                break
            page += 1

        self._category_cache = mapping
        return mapping

    @staticmethod
    def clean_html_to_text(raw_html: str) -> str:
        """
        Strip HTML tags, embedded scripts, styles, and excessive whitespace into
        clean plain text suitable for token counting and embedding.

        Keeps inline tags (<a>, <b>, <em>, <code>) on the same line while inserting
        paragraph breaks after block-level elements (<p>, headings, <li>, etc.).
        """
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "lxml")

        # Remove non-content elements before extracting text.
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        for br in soup.find_all("br"):
            br.replace_with("\n")

        for block in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "tr"]):
            block.append("\n\n")

        raw_text = soup.get_text()
        # Normalize horizontal whitespace inside each line while keeping paragraph breaks.
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
        cleaned_lines = [line for line in lines if line]
        return "\n\n".join(cleaned_lines)

    @staticmethod
    def _parse_wp_datetime(dt_str: str | None) -> datetime | None:
        """Parse WordPress ISO-8601 GMT timestamp strings cleanly."""
        if not dt_str:
            return None
        normalized = dt_str.strip()
        if not normalized:
            return None
        # WordPress `date_gmt` omits the trailing 'Z'; attach UTC offset when absent.
        if normalized.endswith("Z"):
            return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            from datetime import timezone

            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _normalize_post(
        self,
        post: dict[str, Any],
        categories: dict[int, str],
    ) -> dict[str, Any]:
        """Transform a raw WP REST API post object into our SQLite-ready dictionary."""
        wp_post_id = int(post["id"])
        raw_title = post.get("title", {}).get("rendered", "")
        title = html.unescape(BeautifulSoup(raw_title, "lxml").get_text()).strip()

        raw_content = post.get("content", {}).get("rendered", "")
        cleaned_content = self.clean_html_to_text(raw_content)

        cat_ids = post.get("categories") or []
        topic = "General"
        for cid in cat_ids:
            if int(cid) in categories:
                topic = categories[int(cid)]
                break

        return {
            "wp_post_id": wp_post_id,
            "title": title or f"Wololo Post #{wp_post_id}",
            "url": str(post.get("link", "")).strip(),
            "topic": topic,
            "published_at": self._parse_wp_datetime(post.get("date_gmt") or post.get("date")),
            "remote_modified_at": self._parse_wp_datetime(
                post.get("modified_gmt") or post.get("modified")
            ),
            "cleaned_content": cleaned_content,
        }

    def fetch_posts_batch(self, count: int, offset: int = 0) -> list[dict[str, Any]]:
        """
        Fetch `count` posts starting at `offset` (newest first).
        Automatically paginates in chunks of up to 100 (WordPress REST API max per_page).
        """
        if count <= 0:
            return []

        categories = self.fetch_categories()
        collected: list[dict[str, Any]] = []
        current_offset = max(0, offset)

        while len(collected) < count:
            remaining = count - len(collected)
            per_page = min(100, remaining)
            resp = self.session.get(
                self._endpoint("/wp-json/wp/v2/posts"),
                params={
                    "per_page": per_page,
                    "offset": current_offset,
                    "orderby": "date",
                    "order": "desc",
                    "_fields": "id,link,date_gmt,modified_gmt,title,content,categories",
                },
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break

            for raw_post in batch:
                collected.append(self._normalize_post(raw_post, categories))

            current_offset += len(batch)
            if len(batch) < per_page:
                break

        return collected

    def discover_new_posts(
        self,
        target_new_count: int,
        existing_ids: set[int],
        max_pages_to_scan: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Smart Phase-1 discovery helper:
        Scans from newest articles downward so any newly published or edited
        posts are captured, and keeps paging until `target_new_count` unseen
        articles have been collected (or `max_pages_to_scan` pages are scanned).
        """
        if target_new_count <= 0:
            return []

        categories = self.fetch_categories()
        collected: list[dict[str, Any]] = []
        new_found = 0
        page_size = min(100, max(20, target_new_count))

        for page_idx in range(max_pages_to_scan):
            offset = page_idx * page_size
            resp = self.session.get(
                self._endpoint("/wp-json/wp/v2/posts"),
                params={
                    "per_page": page_size,
                    "offset": offset,
                    "orderby": "date",
                    "order": "desc",
                    "_fields": "id,link,date_gmt,modified_gmt,title,content,categories",
                },
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break

            for raw_post in batch:
                normalized = self._normalize_post(raw_post, categories)
                collected.append(normalized)
                if normalized["wp_post_id"] not in existing_ids:
                    new_found += 1
                    if new_found >= target_new_count:
                        return collected

            if len(batch) < page_size:
                break

        return collected
