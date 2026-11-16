"""
WololoRAG root package.

This project is structured as a modular educational RAG (Retrieval-Augmented Generation)
system with two microservices sharing a common core library:
- src/core: Shared Pydantic config, SQLite state DB, Qdrant vector DB, and LangGraph graphs.
- src/services/indexer: CLI worker for two-phase ingestion from Wololo.net.
- src/services/mcp_server: Standalone MCP microservice for LLM clients (Claude/Gemini).
"""
