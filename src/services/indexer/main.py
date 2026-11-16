"""
Indexer Microservice CLI & Worker (src/services/indexer/main.py).

Supports BOTH:
1. Interactive Educational Mode (`python -m src.services.indexer.main`):
   - Runs startup Init diagnostics (compares Wololo.net remote total vs SQLite vs Qdrant).
   - Displays exact counts of DISCOVERED, INDEXED, OUTDATED, and FAILED articles.
   - Lets you choose how many topics to discover into SQL (Phase 1) or vectorize into
     Qdrant (Phase 2), inspect `error_str` failures, and test RAG search queries.
2. Scriptable CLI Flags (`--sync 20 --vectorize 10 --search "PS5 exploit"`):
   - Ideal for Docker jobs, automated testing, or cron-style batch runs.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from langchain_core.embeddings import Embeddings
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from src.core.config import FullSettings, load_settings
from src.core.db.models import ArticleRecord
from src.core.db.repository import ArticleRepository, DatabaseStats
from src.core.graphs.ingestion import run_ingestion_batch
from src.core.graphs.retrieval import run_retrieval_query
from src.core.vector.embeddings import create_embedding_model, get_collection_name
from src.core.vector.qdrant_store import QdrantStoreManager
from src.services.indexer.wololo_client import WololoClient


console = Console()


class IndexerApp:
    """Orchestrates startup diagnostics, Phase 1 SQL sync, Phase 2 LangGraph vectorization, and RAG test queries."""

    def __init__(self, settings: FullSettings) -> None:
        self.settings = settings
        self.active_provider = settings.app.embedding.provider
        self.collection_name = get_collection_name(settings.app)

        self.repo = ArticleRepository(settings.app.database.sqlalchemy_url)
        self.qdrant = QdrantStoreManager(
            app_config=settings.app,
            vector_config=settings.vector,
            collection_name=self.collection_name,
        )
        self.wololo = WololoClient(settings.app.scraper)

        # Lazy-loaded so `--status` or Phase 1 SQL discovery don't wait for PyTorch weights.
        self._embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        """Load the embedding model on first vector operation (cached thereafter)."""
        if self._embeddings is None:
            console.print(
                f"[cyan]Loading embedding provider '{self.active_provider}'...[/cyan]"
            )
            self._embeddings = create_embedding_model(self.settings.app)
        return self._embeddings

    def close(self) -> None:
        """Release Qdrant file locks / HTTP connections."""
        self.qdrant.close()

    def print_init_dashboard(self) -> tuple[Optional[int], DatabaseStats]:
        """
        Startup Init routine:
        Queries Wololo.net total post count, SQLite status breakdown, and Qdrant vector count,
        then displays a unified status panel + recommendation.
        """
        stats = self.repo.get_stats(self.active_provider)
        qdrant_vectors = self.qdrant.count_vectors()

        remote_total: Optional[int] = None
        remote_label = "[yellow]Unavailable (offline/error)[/yellow]"
        try:
            remote_total = self.wololo.get_remote_total_posts()
            not_yet_synced = max(0, remote_total - stats.total_in_sql)
            remote_label = f"[bold white]{remote_total}[/bold white] (not yet in SQL: {not_yet_synced})"
        except Exception as exc:
            remote_label = f"[yellow]Unavailable ({type(exc).__name__})[/yellow]"

        # Model display name
        if self.active_provider == "local":
            model_name = self.settings.app.embedding.local.model_name
        else:
            model_name = self.settings.app.embedding.mistral.model_name

        chunk_cfg = self.settings.vector.chunking
        chunk_desc = (
            f"ON ({chunk_cfg.chunk_size_tokens} tokens, overlap {chunk_cfg.chunk_overlap_tokens})"
            if chunk_cfg.enabled
            else "OFF (1 vector per full article)"
        )

        # Config summary panel
        cfg_summary = (
            f"[bold]Embedding Provider:[/bold] [green]{self.active_provider}[/green] ({model_name})\n"
            f"[bold]SQL Backend:[/bold] {self.repo.backend_name} "
            f"| [bold]Qdrant Mode:[/bold] {self.settings.app.database.qdrant_mode} "
            f"| [bold]Collection:[/bold] [cyan]{self.collection_name}[/cyan]\n"
            f"[bold]Vector Chunking:[/bold] {chunk_desc} "
            f"| [bold]Top-K:[/bold] {self.settings.vector.search.top_k}"
        )
        console.print(Panel(cfg_summary, title="WololoRAG Configuration", border_style="blue"))

        # Database status table
        table = Table(title="Startup Sync Diagnostics (Wololo.net -> SQLite -> Qdrant)")
        table.add_column("Layer / Metric", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Meaning")

        table.add_row("Wololo.net Remote Posts", remote_label, "Total published articles on site")
        table.add_row(
            "SQLite Total Tracked",
            str(stats.total_in_sql),
            "Topics/articles registered in local SQL DB",
        )
        table.add_row(
            "  ├─ DISCOVERED (Pending)",
            f"[yellow]{stats.discovered_count}[/yellow]",
            "In SQLite, waiting to be vectorized into Qdrant",
        )
        table.add_row(
            "  ├─ INDEXED (Active Provider)",
            f"[green]{stats.indexed_for_active_provider}[/green]",
            f"Vectorized with '{self.active_provider}' and up-to-date",
        )
        if stats.indexed_other_provider > 0:
            table.add_row(
                "  ├─ INDEXED (Other Provider)",
                f"[cyan]{stats.indexed_other_provider}[/cyan]",
                "Indexed under the other embedding provider",
            )
        table.add_row(
            "  ├─ OUTDATED",
            f"[magenta]{stats.outdated_count}[/magenta]",
            "Modified on Wololo.net; needs re-vectorization",
        )
        table.add_row(
            "  └─ FAILED (error_str set)",
            f"[red]{stats.failed_count}[/red]",
            "Errored during processing; inspect error_str or retry",
        )
        table.add_row(
            "Qdrant Vectors Stored",
            f"[bold cyan]{qdrant_vectors}[/bold cyan]",
            f"Total chunk points in '{self.collection_name}'",
        )
        console.print(table)

        # Helpful recommendation
        pending = stats.pending_for_vectorization
        if stats.total_in_sql == 0:
            console.print(
                f"[bold yellow]Suggestion:[/bold yellow] Your SQL database is empty. "
                f"Start with Phase 1 to fetch ~{self.settings.app.scraper.default_discover_batch} "
                f"articles into SQLite, then vectorize a batch into Qdrant."
            )
        elif pending > 0:
            console.print(
                f"[bold green]Suggestion:[/bold green] You have {pending} topic(s) in SQLite "
                f"ready to vectorize into Qdrant for provider '{self.active_provider}'."
            )
        else:
            console.print(
                "[bold green]Status:[/bold green] All known SQL articles are synced! "
                "You can fetch more topics from Wololo.net or test RAG search."
            )

        return remote_total, stats

    def sync_catalog(self, count: int) -> dict[str, int]:
        """Phase 1: Pull `count` new articles from Wololo.net into SQLite (DISCOVERED)."""
        if count <= 0:
            return {"inserted": 0, "updated_to_outdated": 0, "unchanged": 0}

        existing_ids = self.repo.get_existing_wp_ids()
        console.print(f"[cyan]Fetching up to {count} new articles from Wololo.net...[/cyan]")
        posts = self.wololo.discover_new_posts(
            target_new_count=count,
            existing_ids=existing_ids,
        )
        summary = self.repo.upsert_discovered_articles(posts)
        console.print(
            f"[green]Phase 1 Complete:[/green] Inserted={summary['inserted']}, "
            f"Marked OUTDATED={summary['updated_to_outdated']}, "
            f"Unchanged={summary['unchanged']}"
        )
        return summary

    def vectorize_pending(self, count: int, include_failed: bool = False) -> None:
        """Phase 2: Run up to `count` pending SQLite articles through LangGraph -> Qdrant."""
        if count <= 0:
            return

        pending = self.repo.get_pending_for_vectorization(
            limit=count,
            active_provider=self.active_provider,
            include_failed=include_failed,
        )
        if not pending:
            console.print("[yellow]No pending articles found in SQLite to vectorize.[/yellow]")
            return

        embeddings = self.get_embeddings()
        console.print(
            f"[cyan]Running LangGraph Ingestion Pipeline for {len(pending)} article(s)...[/cyan]"
        )

        def _log_step(idx: int, total: int, art: ArticleRecord, status: str, chunks: int) -> None:
            color = "green" if status == "INDEXED" else "red"
            console.print(
                f"  [{idx}/{total}] Post #{art.wp_post_id} [{art.topic}] "
                f"'{art.title[:55]}' -> [{color}]{status}[/{color}] ({chunks} chunks)"
            )

        result = run_ingestion_batch(
            articles=pending,
            settings=self.settings,
            repo=self.repo,
            embeddings=embeddings,
            qdrant_store=self.qdrant,
            on_progress=_log_step,
        )

        console.print(
            f"[bold green]Phase 2 Complete:[/bold green] "
            f"Succeeded={result.succeeded}, Failed={result.failed}, "
            f"Chunks Written={result.total_chunks_written}"
        )
        if result.errors:
            console.print("[bold red]Errors recorded in SQLite (`error_str`):[/bold red]")
            for post_id, title, err in result.errors:
                console.print(f"  - Post #{post_id} ({title[:40]}): {err}")

    def inspect_and_retry_failed(self, retry_limit: int = 10) -> None:
        """Show FAILED articles with `error_str` and optionally retry them."""
        failed_rows = self.repo.get_failed_articles(limit=retry_limit)
        if not failed_rows:
            console.print("[green]No FAILED articles in SQLite![/green]")
            return

        table = Table(title="Failed Articles (SQLite `error_str`)")
        table.add_column("WP ID", justify="right")
        table.add_column("Topic")
        table.add_column("Title")
        table.add_column("error_str", style="red")
        for row in failed_rows:
            table.add_row(
                str(row.wp_post_id),
                row.topic,
                row.title[:40],
                (row.error_str or "")[:120],
            )
        console.print(table)

    def run_search(
        self,
        query: str,
        topic_filter: str | None = None,
        top_k: int | None = None,
        published_after: str | None = None,
    ) -> None:
        """Execute LangGraph Retrieval pipeline and display matched chunks."""
        embeddings = self.get_embeddings()
        state = run_retrieval_query(
            query=query,
            settings=self.settings,
            repo=self.repo,
            embeddings=embeddings,
            qdrant_store=self.qdrant,
            topic_filter=topic_filter,
            top_k=top_k,
            published_after=published_after,
        )
        hits = state.get("enriched_hits", [])
        console.print(
            Panel(
                state.get("formatted_context", ""),
                title=f"RAG Retrieval Results ({len(hits)} matches)",
                border_style="green" if hits else "yellow",
            )
        )

    def clear_databases(self, require_confirmation: bool = True) -> None:
        """Clear all articles in SQL and drop the active Qdrant collection."""
        if require_confirmation:
            confirmed = Confirm.ask(
                "[bold red]Are you sure you want to permanently clear ALL articles in SQL "
                "and ALL vectors in Qdrant?[/bold red]",
                default=False,
            )
            if not confirmed:
                console.print("[yellow]Clear cancelled.[/yellow]")
                return

        deleted_vectors = self.qdrant.clear_collection()
        deleted_rows = self.repo.clear_all_articles()
        console.print(
            f"[bold green]Cleared:[/bold green] Removed {deleted_rows} SQL article(s) "
            f"and {deleted_vectors} Qdrant vector(s) from '{self.collection_name}'."
        )

    def interactive_loop(self) -> None:
        """Interactive console menu."""
        _, stats = self.print_init_dashboard()

        while True:
            console.print("\n[bold cyan]Choose an action:[/bold cyan]")
            console.print("  [1] Phase 1: Sync/Discover new articles from Wololo.net -> SQL")
            console.print("  [2] Phase 2: Vectorize pending articles from SQL -> Qdrant (LangGraph)")
            console.print("  [3] Inspect & Retry FAILED articles (`error_str`)")
            console.print("  [4] Test Semantic RAG Search (LangGraph Retrieval)")
            console.print("  [5] Refresh Startup Status Dashboard")
            console.print("  [6] Clear Databases (SQL + Qdrant) [with confirmation]")
            console.print("  [0] Exit")

            choice = Prompt.ask(
                "Select option",
                choices=["0", "1", "2", "3", "4", "5", "6"],
                default="2",
            )

            if choice == "0":
                console.print("[cyan]Goodbye![/cyan]")
                break
            elif choice == "1":
                default_batch = self.settings.app.scraper.default_discover_batch
                n = IntPrompt.ask(
                    "How many NEW articles to fetch into SQL?",
                    default=default_batch,
                )
                self.sync_catalog(n)
            elif choice == "2":
                stats = self.repo.get_stats(self.active_provider)
                default_vec = min(10, max(1, stats.pending_for_vectorization))
                n = IntPrompt.ask(
                    f"How many pending articles to vectorize now? (available: {stats.pending_for_vectorization})",
                    default=default_vec,
                )
                self.vectorize_pending(n)
            elif choice == "3":
                self.inspect_and_retry_failed()
                stats = self.repo.get_stats(self.active_provider)
                if stats.failed_count > 0 and Confirm.ask("Retry all failed articles now?", default=False):
                    self.vectorize_pending(stats.failed_count, include_failed=True)
            elif choice == "4":
                q = Prompt.ask("Enter your search question")
                topic = Prompt.ask("Optional topic filter (leave blank for all)", default="")
                after = Prompt.ask("Optional published_after YYYY-MM-DD (leave blank for all)", default="")
                self.run_search(
                    q,
                    topic_filter=topic or None,
                    published_after=after or None,
                )
            elif choice == "5":
                _, stats = self.print_init_dashboard()
            elif choice == "6":
                self.clear_databases(require_confirmation=True)
                _, stats = self.print_init_dashboard()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WololoRAG Indexer: Two-phase ingestion & RAG testing CLI."
    )
    parser.add_argument("--app-config", type=str, default=None, help="Path to app_config.yaml")
    parser.add_argument("--vector-config", type=str, default=None, help="Path to vector_config.yaml")
    parser.add_argument("--status", action="store_true", help="Print startup DB diagnostics and exit")
    parser.add_argument("--sync", type=int, default=None, help="Phase 1: Sync N new articles into SQL")
    parser.add_argument("--vectorize", type=int, default=None, help="Phase 2: Vectorize N pending SQL articles into Qdrant")
    parser.add_argument("--retry-failed", type=int, default=None, help="Retry up to N FAILED articles")
    parser.add_argument("--search", type=str, default=None, help="Run a semantic RAG query")
    parser.add_argument("--topic", type=str, default=None, help="Optional topic filter for --search")
    parser.add_argument("--published-after", type=str, default=None, help="Optional YYYY-MM-DD filter for --search")
    parser.add_argument("--top-k", type=int, default=None, help="Override search top_k")
    parser.add_argument("--clear", action="store_true", help="Clear all SQL rows and Qdrant vectors")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt when using --clear")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(
        app_config_path=args.app_config,
        vector_config_path=args.vector_config,
    )
    app = IndexerApp(settings)
    try:
        has_cli_actions = any(
            [
                args.status,
                args.clear,
                args.sync is not None,
                args.vectorize is not None,
                args.retry_failed is not None,
                args.search is not None,
            ]
        )

        if not has_cli_actions:
            app.interactive_loop()
            return 0

        app.print_init_dashboard()
        if args.clear:
            app.clear_databases(require_confirmation=not args.yes)
        if args.sync is not None:
            app.sync_catalog(args.sync)
        if args.vectorize is not None:
            app.vectorize_pending(args.vectorize)
        if args.retry_failed is not None:
            app.inspect_and_retry_failed(args.retry_failed)
            app.vectorize_pending(args.retry_failed, include_failed=True)
        if args.search is not None:
            app.run_search(
                args.search,
                topic_filter=args.topic,
                top_k=args.top_k,
                published_after=args.published_after,
            )

        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
