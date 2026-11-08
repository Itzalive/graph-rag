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
fact), `FactSearchRequest` (a query vector, a limit of at most 100, ids to exclude, (id, seq) pairs to exclude only at that version (`exclude_versions`, a missing seq counting as 0), and whether to include expired facts
(`include_expired`) and not-yet-valid ones (`include_future`)) and `FactSearchHit` (a fact with its score, a cosine distance: 0 is identical, lower is closer; hits come closest
first). Two adapters implement it:

- `graph_rag.adapters.in_memory_facts.InMemoryFactBackend`: exact cosine search over facts held in memory, for tests and
  evals.
- `graph_rag.adapters.cosmos.CosmosFactBackend`: the Azure Cosmos DB `memory` container, one family per backend
  (`family_id`), vector search through `VectorDistance`. With the cosine function Cosmos returns a similarity (1 is
  identical), so the adapter reports 1 minus it.

`FactRecord` and `FactSearchHit` also carry the recorded-memory fields (`kind`, `holder` as a `Holder(kind, ref)`,
`lifespan`, `valid_from`, `validity_basis`, `forgotten_at`, `purged_at`, `learned_at`, `last_changed_at`,
`last_confirmed_at`, `about` (entity ids the memory is about), `mentions` (other entity ids it names), `topics`,
`seq`, `situation` (a `Situation`, see "Situations and events") and `constraint`); all are optional and a document written before 0.3.0 reads as
`kind None`, `seq 0` (and a document without `situation` or `constraint` reads `None` and `False`). List and search return only current memories: never deleted, forgotten or purged; past
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

- `write_change(record, changes, entities=(), entity_updates=())` writes the memory, its one or two `FactChange` records
  (two only for an old document's first change, the first at `seq 0` carrying its own `after`), an `op-<op_id>` marker,
  any new entities and any entity updates in one transactional batch. An entity update replaces the entity only if it is
  unchanged since it was read: it must carry the `version` that read returned (Cosmos: a batch `replace` with
  `if_match_etag`). The backend sets the last change's `after` to the memory's state. Before sending it refuses
  (`ValueError`) a write Cosmos could not take: more than 100 batch operations (the marker, the changes, new entities and entity updates with a change record each, and the memory), an entity id twice, an entity update without a version, or a `/`, `\`, `?` or `#` in
  the fact id or op id. It then runs the write checks W1–W4 (`WriteCheckError(code)`): a dated
  memory has a `valid_until`; a `valid_until` on any other lifespan has basis `ended`; a validity date has a basis; the
  memory's `seq` is the last change's.
- A change record's id is `<fact_id>:<seq>`. A writer reads the memory's `seq` and writes `seq + 1`; if another write
  took it, created an entity this write creates, or changed or removed an entity this write updates, `StaleWriteError` is raised; nothing is written: re-read and apply once more.
- `op_id` names the request. If its marker exists the request was applied before: `AlreadyAppliedError`, which callers
  treat as success with nothing written. Markers are kept. `op_applied(op_id)` says whether a change with that op id
  was written (its marker exists in the bucket; on Cosmos one point read), so a caller can skip work a repeated
  request would only throw away. A marker records the `seq` its write left: `op_marker(op_id)` returns `OpMarker(seq)`
  for a written marker (`seq` None for a marker that has no `seq`) or None when it was never written, in the same one
  point read, so a repeated request learns the version it left.
- `history(fact_id, *, after_seq=-1, limit=100)` returns change records in `seq` order, at most `limit` (1–500); page
  by passing the last `seq` received.
- `set_vectors(fact_id, vectors, *, expected_seq)` fills a memory's vectors only if it still has `expected_seq` (and, on
  Cosmos, its etag is unchanged); it returns whether it wrote and writes no change record.
- `situational(channel, types, entity_ids, *, unsituated_kinds=(), limit=100)` reads current memories twice: those
  whose situation applies (`situation_applies`), then, only when `unsituated_kinds` is not empty, those of one of those
  kinds that have no situation; the result is the first rows then the second. Each read is ordered by a true
  `constraint` first, then `learned_at` as the stored string (a missing one first), then id, and keeps at most `limit`
  rows (1–100), raising `situational.limit_reached` when it drops any. `channel` must be non-empty and `types` and
  `entity_ids` hold at most 200 distinct values each; otherwise `ValueError` before any read. The records carry no
  vectors; on Cosmos only the search's named fields reach `extras`. A result is for reading only: to write a memory
  found here, read it with `read_fact` first, or its vectors and unlisted extras are lost. On Cosmos it needs the
  composite index `(/constraint DESC, /learned_at ASC, /id ASC)`. A string passed for `types`, `entity_ids` or
  `unsituated_kinds` (or for `find_by_aliases`' `keys`) is a `TypeError`: each takes a collection of strings.

### Entities

`graph_rag.entities.EntityRecord(entity_id, type, name, handle, aliases, identifiers, seq, version, merged_into)` lives in the same
bucket (`doc_type: "entity"`). `handle` is required: the caller's readable id for the entity; it is not an alias key.
`version` is set by every read (Cosmos: the document's `_etag`; in memory: a counter), moves on every write, is never
stored and is not part of equality. An entity document without a handle raises
`MissingHandleError` (a `ValueError` naming the entity id) on every read: reset the store that holds it. `upsert_entity`, `read_entity`, `read_entities(ids)` (the entities with those ids in
one query, sorted by id, missing ids skipped; at most 200 distinct ids, `ValueError` above), `list_entities(type=None)`,
`find_by_alias(type, alias)` (exact, case-insensitive, over the name and aliases) and `find_by_identifier(kind, value)`
(case-insensitive value). Types, handles and ids are the caller's.
- `merged_into` is the id of the entity this one was merged into; it is stored only when set. What it means to readers
  is the caller's.
- `find_by_aliases(keys, *, limit=50)` returns entities of any type whose name or aliases match one of `keys`, in one
  query. Keys are folded as `find_by_alias` folds them, empty ones dropped and repeats removed in the caller's order;
  past 300 distinct keys only the first 300 are used (`aliases.keys_capped`); no keys returns `[]` without a read. At
  most 300 matching entities are read (`aliases.read_capped` past them; in memory, the 300 with the lowest ids).
  Results are ranked by the longest matching key in characters, then id, and cut to `limit` (1–50, otherwise
  `ValueError`; `aliases.limit_reached` when it cuts).
- `replace_entities(updates, *, source, at)` replaces each entity only if its stored version is still the `version`
  the update carries, writes it at the update's `seq` plus one, and writes one entity change record for it; all apply
  or none does (on Cosmos, one transactional batch: the conditional `replace` operations, then a `create` per record).
  An update whose entity moved on or is gone, or whose record id is already taken, raises `StaleWriteError` naming
  that entity's id, and nothing is written; any other Cosmos failure is raised unchanged. A repeat of an applied call
  carries the version the first one replaced, so it is stale and changes nothing: re-read and decide again. A call
  takes at most 50 updates with distinct ids, each carrying a version (`ValueError` before anything is sent); no
  updates sends nothing.
- Entity history (0.5.0): every entity `write_change` creates (written at `seq` 1) or updates (at `seq` plus one), and
  every `replace_entities` update, gets an `EntityChange(entity_id, seq, kind, at, source, after)` in the same batch:
  `kind` `created` or `changed`, `source` and `at` the memory change's (its last) or the ones `replace_entities` was
  given, `after` the entity's state (type, name, handle, aliases, identifiers, seq, `merged_into` when set). The record
  is a document of its own (`doc_type: "entity_change"`, id `entity_change:<entity id>:<seq>`). The backend sets `seq`,
  never the caller; `upsert_entity` stores the `seq` it is given and writes no record.
  `entity_history(entity_id, *, after_seq=-1, limit=100)` returns the records in `seq` order (`limit` 1–500).

### Situations and events

`graph_rag.facts.Situation(channels, types=(), entities=(), topics=())` says where a memory applies: the channels it
applies in and, optionally, the input types, entity ids and topics it is limited to. `FactRecord` and `FactSearchHit`
carry it as `situation`; the stored document always holds `channels` and leaves out the other lists when empty; a missing key reads as empty, and any other key is ignored on read and not written back. `constraint` is a bool that is stored only when
true. A caller sets `situation` and `constraint` on the record itself, never through `extras`: an extras value is written
as given, so the two backends would read it differently. `check_write` refuses a situation without a channel
(`ValueError`). `situation_applies(situation, channel, types, entity_ids)` is the matching rule: the channel is one of the situation's `channels`; its `types`, when it has any, share a value with `types`; its `entities`, when it has any, share one with `entity_ids`; and it has no `topics`.

Both adapters take `events=` (an `EventSink`, `graph_rag.events.EventSink`; `NO_EVENTS` unless one is passed) and call
`count(event, **fields)` on it for four events:

- `situational.limit_reached` with `query` and `limit`;
- `aliases.keys_capped` with `given` and `used`;
- `aliases.read_capped`;
- `aliases.limit_reached` with `limit`.

The fields are integers or graph-rag's fixed words only (`query` is `situated` or `unsituated`).
