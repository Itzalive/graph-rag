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

`graph_rag.facts` defines the `FactBackend` protocol (`upsert_fact`, `read_fact`,
`list_facts(*, include_expired=False, include_future=False)`, `search_facts`) and its records: `FactRecord` (a stored
fact), `FactSearchRequest` (a query vector, a limit of at most 100, ids to exclude, and whether to include expired facts
(`include_expired`) and not-yet-valid ones (`include_future`)) and `FactSearchHit` (a fact with its score, a cosine distance: 0 is identical, lower is closer; hits come closest
first). Two adapters implement it:

- `graph_rag.adapters.in_memory_facts.InMemoryFactBackend`: exact cosine search over facts held in memory, for tests and
  evals.
- `graph_rag.adapters.cosmos.CosmosFactBackend`: the Azure Cosmos DB `memory` container, one family per backend
  (`family_id`), vector search through `VectorDistance`. With the cosine function Cosmos returns a similarity (1 is
  identical), so the adapter reports 1 minus it.

`FactRecord` and `FactSearchHit` also carry the recorded-memory fields (`kind`, `holder` as a `Holder(kind, ref)`,
`lifespan`, `valid_from`, `validity_basis`, `forgotten_at`, `purged_at`, `learned_at`, `last_changed_at`,
`last_confirmed_at`, `entities`, `topics`, `seq`); all are optional and a document written before 0.3.0 reads as
`kind None`, `seq 0`. List and search return only current memories: never deleted, forgotten or purged; past
`valid_until` only with `include_expired`; before `valid_from` only with `include_future`; documents with a `doc_type`
(change records, op markers, entities) are never memories.

### The clock

Both adapters take an optional keyword argument `clock` (`graph_rag.clock.Clock`: a no-argument callable returning a
`datetime`). It decides "now" wherever an adapter needs it: the `valid_until` and `valid_from` cut-offs of list and search
(a fact whose `valid_until` is at or before now is expired) and, in the Cosmos adapter, the `created_at` / `updated_at` of
a stored document that lacks them. The default, `graph_rag.clock.utc_now`, reads the system clock. The clock must return
an aware datetime: `graph_rag.clock.read_clock` raises `ValueError` for a naive one, since stored times are compared as
aware UTC values.

### Buckets

`CosmosFactBackend(..., family_id, scope="shared")` keeps one bucket, the partition `[family_id, scope, "FACTS"]`, on
every call; `InMemoryFactBackend` holds one bucket per instance.

### The record

`graph_rag.record.RecordedFactBackend` extends `FactBackend`; both adapters implement it.

- `write_change(record, changes, entities=())` writes the memory, its one or two `FactChange` records (two only for an
  old document's first change, the first at `seq 0` carrying its own `after`), an `op-<op_id>` marker and any new
  entities in one transactional batch. The backend sets the last change's `after` to the memory's state. Before sending
  it refuses (`ValueError`) a write Cosmos could not take: more than 100 batch operations, an entity id twice, or a
  `/`, `\`, `?` or `#` in the fact id or op id. It then runs the write checks W1–W4 (`WriteCheckError(code)`): a dated
  memory has a `valid_until`; a `valid_until` on any other lifespan has basis `ended`; a validity date has a basis; the
  memory's `seq` is the last change's.
- A change record's id is `<fact_id>:<seq>`. A writer reads the memory's `seq` and writes `seq + 1`; if another write
  took it, or created an entity this write creates, `StaleWriteError` is raised: re-read and apply once more.
- `op_id` names the request. If its marker exists the request was applied before: `AlreadyAppliedError`, which callers
  treat as success with nothing written. Markers are kept.
- `history(fact_id, *, after_seq=-1, limit=100)` returns change records in `seq` order, at most `limit` (1–500); page
  by passing the last `seq` received.
- `set_vectors(fact_id, vectors, *, expected_seq)` fills a memory's vectors only if it still has `expected_seq` (and, on
  Cosmos, its etag is unchanged); it returns whether it wrote and writes no change record.

### Entities

`graph_rag.entities.EntityRecord(entity_id, type, name, aliases, identifiers, seq)` lives in the same bucket
(`doc_type: "entity"`). `upsert_entity`, `read_entity`, `list_entities(type=None)`, `find_by_alias(type, alias)` (exact,
case-insensitive, over the name and aliases) and `find_by_identifier(kind, value)` (case-insensitive value). Types and
ids are the caller's.
