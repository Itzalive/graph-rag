# graph-rag

Library-first bootstrap repository for the family fact-store extraction roadmap.

## Local bootstrap commands

uv sync
.venv/bin/pytest -q
python -m graph_rag check

## Local backends

- `GRAPH_RAG_BACKEND=sqlite` stores facts in `data/graph_rag.sqlite3`
- `GRAPH_RAG_BACKEND=memory` uses the in-memory adapter for local contract checks

## Shard boundary

This bootstrap shard is intentionally local-only. It does not include cloud adapters,
live assistant extraction, or MCP server implementation.
