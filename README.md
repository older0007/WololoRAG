# WololoRAG — Indexer Worker (Write Service)

An educational, production-structured **RAG (Retrieval-Augmented Generation)** ingestion service that fetches articles from **Wololo.net**, tracks synchronization state in **SQL (PostgreSQL or SQLite)**, splits text by LLM tokens, generates vector embeddings (**Local HuggingFace** or **Mistral API**), and stores them in **Qdrant** via **LangChain**, **LangGraph**, and **Pydantic**.

---

## 3-Repository Architecture (CQRS Separation)

| Repository | Responsibility | GitHub |
| :--- | :--- | :--- |
| **`WololoRAG-Infra`** | Shared Docker infrastructure (`PostgreSQL` + `Adminer` + `Qdrant`) | https://github.com/older0007/WololoRAG-Infra |
| **`WololoRAG`** *(this repo)* | **Indexer Worker (Write Side)**: crawls Wololo.net, chunks text, embeds & writes vectors + SQL state | https://github.com/older0007/WololoRAG |
| **`WololoRAG-MCP`** | **MCP Server (Read Side)**: exposes search & metadata tools to LLMs (Antigravity, Claude, Gemini) | https://github.com/older0007/WololoRAG-MCP |

---

## Key Features

- **Pydantic Configuration Validation**:
  - [`config/app_config.yaml`](config/app_config.yaml): switch between `local` (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) and `mistral` (`mistral-embed`, 1024-dim), configure SQL/Qdrant targets, and separate vector collections automatically (`wololo_articles_local` vs `wololo_articles_mistral`).
  - [`config/vector_config.yaml`](config/vector_config.yaml): dedicated vector settings (`chunking.enabled` toggle, `chunk_size_tokens`, `chunk_overlap_tokens`, `top_k`, `score_threshold`, `distance_metric`).
- **Two-Phase Ingestion Pipeline (LangGraph `StateGraph`)**:
  1. **Phase 1 — Catalog Sync**: fetches article metadata and cleaned plain text from Wololo.net's WordPress REST API into SQL (`status = DISCOVERED`, or marks modified indexed articles as `OUTDATED`).
  2. **Phase 2 — Batch Vectorization**: processes user-selected batches (`N` topics) through a conditional LangGraph workflow (`validate_and_prepare_text` -> `chunk_by_tokens` / `single_document_chunk` -> `embed_and_store_qdrant` -> `commit_sql_status`), recording exact Qdrant `vector_ids` or capturing exceptions into `error_str` (`status = FAILED`).
- **Startup Diagnostics**: inspects total posts on Wololo.net (`X-WP-Total`), local SQL status breakdown (`DISCOVERED`, `INDEXED`, `OUTDATED`, `FAILED`), and Qdrant vector counts on every launch.

---

## Setup & Usage

### 1. Environment Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

### 2. Choose Database Backend

#### Option A: Shared Docker Databases via `WololoRAG-Infra` (PostgreSQL + Qdrant on `localhost`)

1. Start the database stack in `WololoRAG-Infra`:
   ```bash
   cd ../WololoRAG-Infra
   docker compose up -d
   ```
2. Copy `.env.example` to `.env` in this repository and uncomment:
   ```env
   DATABASE_URL=postgresql+psycopg://wololo:wololo_pass@localhost:5432/wololo_rag
   QDRANT_MODE=server
   QDRANT_URL=http://localhost:6333
   ```

#### Option B: Local File Databases (Default when `.env` is empty)
Stores SQL state in `./data/wololo_state.db` (SQLite) and vectors in `./data/qdrant`.

### 3. Run the Indexer

```bash
# Interactive mode (startup diagnostics + step-by-step menu)
.venv/bin/python -m src.services.indexer.main

# Scripted CLI flags
.venv/bin/python -m src.services.indexer.main --status
.venv/bin/python -m src.services.indexer.main --sync 20 --vectorize 10 --search "PS5 exploit"
```

---

## Tests

```bash
.venv/bin/pytest -v
```
