# graph-rag

Library-first bootstrap repository for the family fact-store extraction roadmap.

## Local bootstrap commands

```bash
uv sync
.venv/bin/pytest -q
.venv/bin/python -m graph_rag check
```

## Local backends

- `GRAPH_RAG_BACKEND=sqlite` stores facts in `data/graph_rag.sqlite3`
- `GRAPH_RAG_BACKEND=memory` uses the in-memory adapter for local contract checks

## Fact backends

`graph_rag.facts` defines the `FactBackend` protocol (`upsert_fact`, `read_fact`, `list_facts`, `search_facts`) and its
records: `FactRecord` (a stored fact), `FactSearchRequest` (a query vector, a limit of at most 100, ids to exclude, and
whether to include expired facts) and `FactSearchHit` (a fact with its score). Two adapters implement it:

- `graph_rag.adapters.in_memory_facts.InMemoryFactBackend`: exact cosine search over facts held in memory, for tests and
  evals.
- `graph_rag.adapters.cosmos.CosmosFactBackend`: the Azure Cosmos DB `memory` container, one family per backend
  (`family_id`), vector search through `VectorDistance`.

### The clock

Both adapters take an optional keyword argument `clock` (`graph_rag.clock.Clock`: a no-argument callable returning a
`datetime`). It decides "now" wherever an adapter needs it: the expiry cut-off (a fact whose `valid_until` is at or before
now is left out unless expired facts are requested) and, in the Cosmos adapter, the `created_at` / `updated_at` of a stored
document that lacks them. The default, `graph_rag.clock.utc_now`, reads the system clock. The clock must return an
aware datetime: `graph_rag.clock.read_clock` raises `ValueError` for a naive one, since stored times are compared as aware
UTC values.
