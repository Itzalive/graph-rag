from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from graph_rag.clock import Clock, read_clock, utc_now
from graph_rag.entities import (
    EntityRecord, alias_key, check_entity_ids, entity_from_state, entity_to_state, identifier_key,
)
from graph_rag.facts import (
    SYSTEM_KEYS, FactRecord, FactSearchHit, FactSearchRequest, fact_state, hit_from_state, record_from_state,
)
from graph_rag.record import (
    HISTORY_DEFAULT_LIMIT, AlreadyAppliedError, FactChange, StaleWriteError, change_from_state, change_to_state,
    check_history_limit, check_write, op_marker_id,
)


# Every fact field a search hit can use, including the extras seen in production facts (claude-assistant-yh9p) and the
# recorded-memory fields with Kerwen's own extras (claude-assistant-xz59.9 §3).
_SEARCH_FIELDS = (
    "id", "fact", "created_at", "last_updated_at", "source", "subject", "fact_type", "confidence", "valid_until",
    "family_id", "user_id", "category", "is_shared", "deleted", "deleted_at",
    "extracted_at", "history", "metadata", "updated_at", "user",
    "kind", "holder", "lifespan", "valid_from", "validity_basis", "forgotten_at", "purged_at", "learned_at",
    "last_changed_at", "last_confirmed_at", "about", "mentions", "topics", "seq",
    "constraint", "situation", "attribute", "calc",
)
_SEARCH_SELECT = ", ".join(f"c.{field}" for field in _SEARCH_FIELDS)

# WHERE clauses for memory reads. tests/cosmos_fake.py interprets each by its exact text: change both together.
CLAUSE_MEMORY = "NOT IS_DEFINED(c.doc_type)"
CLAUSE_NOT_DELETED = "(NOT IS_DEFINED(c.deleted) OR c.deleted != true)"
CLAUSE_NOT_FORGOTTEN = "(NOT IS_DEFINED(c.forgotten_at) OR IS_NULL(c.forgotten_at))"
CLAUSE_NOT_PURGED = "(NOT IS_DEFINED(c.purged_at) OR IS_NULL(c.purged_at))"
CLAUSE_NOT_EXPIRED = "(NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until) OR c.valid_until > @now)"
CLAUSE_STARTED = "(NOT IS_DEFINED(c.valid_from) OR IS_NULL(c.valid_from) OR c.valid_from <= @now)"
CLAUSE_NOT_EXCLUDED = "ARRAY_CONTAINS(@exclude_ids, c.id) = false"
CLAUSE_CHANGES = "c.doc_type = 'fact_change'"
CLAUSE_CHANGE_OF_FACT = "c.fact_id = @fact_id"
CLAUSE_AFTER_SEQ = "c.seq > @after"
CLAUSE_ENTITIES = "c.doc_type = 'entity'"
CLAUSE_ENTITY_TYPE = "c.type = @type"
CLAUSE_ALIAS = "ARRAY_CONTAINS(c.alias_keys, @alias)"
CLAUSE_IDENTIFIER = "ARRAY_CONTAINS(c.identifier_keys, @identifier)"
CLAUSE_ENTITY_IDS = "ARRAY_CONTAINS(@ids, c.id)"

# A transactional batch holds at most 100 operations (Azure Cosmos DB limit).


class CosmosFactBackend:
    """Cosmos DB-backed FactBackend.

    Constructor injection: pass a pre-built ``ContainerProxy`` or supply
    ``cosmos_url``, ``credential``, and ``database_name`` to build one lazily.
    The container name is always ``"memory"`` (the shared assistant container).
    One bucket per backend: ``[family_id, scope, "FACTS"]``, ``scope`` ``"shared"`` unless the caller names another.
    """

    def __init__(
        self,
        container=None,
        *,
        cosmos_url: str | None = None,
        credential=None,
        database_name: str | None = None,
        family_id: str,
        scope: str = "shared",
        clock: Clock = utc_now,
    ) -> None:
        if container is None and (cosmos_url is None or credential is None or database_name is None):
            raise ValueError(
                "Provide either a ContainerProxy or all of: cosmos_url, credential, database_name"
            )
        if not scope:
            raise ValueError("scope must name a bucket, for example 'shared'")
        # A URL-built client is made on first use, inside the worker thread the operation runs in: the CosmosClient
        # constructor fetches the database account over the network (claude-assistant-5si8).
        self._container_client = container
        self._cosmos_url = cosmos_url
        self._credential = credential
        self._database_name = database_name
        self._build_lock = threading.Lock()
        self._family_id = family_id
        self._scope = scope
        self._pk = [family_id, scope, "FACTS"]
        self._clock = clock

    @property
    def _container(self):
        """The container, built once on first use. Blocking: read it only inside a worker thread."""
        if self._container_client is None:
            with self._build_lock:
                if self._container_client is None:
                    from azure.cosmos import CosmosClient  # noqa: PLC0415

                    client = CosmosClient(url=self._cosmos_url, credential=self._credential)
                    self._container_client = client.get_database_client(self._database_name).get_container_client(
                        "memory"
                    )
        return self._container_client

    # --- FactBackend protocol ---

    async def upsert_fact(self, record: FactRecord) -> None:
        doc = self._to_document(record)
        await asyncio.to_thread(lambda: self._container.upsert_item(doc))

    async def read_fact(self, fact_id: str) -> FactRecord | None:
        doc = await self._read_raw(fact_id)
        return None if doc is None or "doc_type" in doc else self._from_document(doc)

    async def list_facts(self, *, include_expired: bool = False, include_future: bool = False) -> list[FactRecord]:
        filters, params = self._memory_filters(include_expired=include_expired, include_future=include_future)
        query = "SELECT * FROM c WHERE " + " AND ".join(filters)
        results = await asyncio.to_thread(
            lambda: list(
                self._container.query_items(
                    query=query,
                    parameters=params or None,
                    partition_key=self._pk,
                )
            )
        )
        return [self._from_document(doc) for doc in results]

    async def search_facts(self, request: FactSearchRequest) -> list[FactSearchHit]:
        if request.limit > 100:
            raise ValueError("limit must not exceed 100")
        filters, params = self._memory_filters(include_expired=request.include_expired,
                                               include_future=request.include_future)
        params.insert(0, {"name": "@qv", "value": list(request.query_vector)})
        if request.exclude_ids:
            filters.append(CLAUSE_NOT_EXCLUDED)
            params.append({"name": "@exclude_ids", "value": list(request.exclude_ids)})
        where_clause = " AND ".join(filters)
        # Named fields, never `c`: each vector_embeddings is ~22 KB of JSON that the hit never uses. A field that is not
        # listed in _SEARCH_FIELDS does not reach FactSearchHit.extras; add it there when a new fact field is introduced.
        query = (
            f"SELECT TOP {request.limit} {_SEARCH_SELECT}, VectorDistance(c.vector_embeddings, @qv, false) AS score "
            f"FROM c WHERE {where_clause} "
            f"ORDER BY VectorDistance(c.vector_embeddings, @qv, false)"
        )
        raw_results = await asyncio.to_thread(
            lambda: list(
                self._container.query_items(
                    query=query,
                    parameters=params,
                    partition_key=self._pk,
                )
            )
        )
        # With the cosine distance function VectorDistance returns a similarity (1 = identical) and ORDER BY puts the most
        # similar first: 1 - similarity is the FactSearchHit cosine distance, already in ascending order.
        return [hit_from_state(doc, 1.0 - doc["score"] if "score" in doc else float("inf")) for doc in raw_results]

    # --- RecordedFactBackend ---

    async def write_change(self, record: FactRecord, changes: Sequence[FactChange],
                           entities: Sequence[EntityRecord] = ()) -> None:
        """Write the memory, its change records, the op marker and any new entities in one transactional batch (spec §2).
        A request whose marker exists raises AlreadyAppliedError before anything is sent; a create conflict on the marker
        inside the batch (two copies of one request at once) raises it too; any other create conflict (a change record's
        seq, or a new entity's id) raises StaleWriteError. Any other batch failure is raised unchanged. The memory
        document is replaced whole: pass the vectors to keep."""
        from azure.cosmos.exceptions import CosmosBatchOperationError  # noqa: PLC0415

        check_write(record, changes, entities)
        op_id = changes[-1].op_id
        marker = op_marker_id(op_id)
        if await self._read_raw(marker) is not None:
            raise AlreadyAppliedError(op_id)
        stored = [*changes[:-1], replace(changes[-1], after=fact_state(record))]
        operations: list[tuple[str, tuple[dict[str, Any]]]] = [
            ("create", (self._stamp({"id": marker, "doc_type": "fact_op", "op_id": op_id, "fact_id": record.fact_id}),)),
            *[("create", (self._stamp(change_to_state(change)),)) for change in stored],
            *[("create", (self._stamp(entity_to_state(entity)),)) for entity in entities],
            ("upsert", (self._to_document(record),)),
        ]
        try:
            await asyncio.to_thread(
                lambda: self._container.execute_item_batch(batch_operations=operations, partition_key=self._pk)
            )
        except CosmosBatchOperationError as exc:
            if exc.status_code == 409 and exc.error_index == 0:
                raise AlreadyAppliedError(op_id) from exc
            if exc.status_code == 409:
                raise StaleWriteError(record.fact_id) from exc
            raise

    async def history(self, fact_id: str, *, after_seq: int = -1,
                      limit: int = HISTORY_DEFAULT_LIMIT) -> list[FactChange]:
        check_history_limit(limit)
        query = (f"SELECT TOP @n * FROM c WHERE {CLAUSE_CHANGES} AND {CLAUSE_CHANGE_OF_FACT} AND {CLAUSE_AFTER_SEQ} "
                 "ORDER BY c.seq ASC")
        params = [{"name": "@n", "value": limit}, {"name": "@fact_id", "value": fact_id},
                  {"name": "@after", "value": after_seq}]
        rows = await asyncio.to_thread(
            lambda: list(self._container.query_items(query=query, parameters=params, partition_key=self._pk))
        )
        return [change_from_state(row) for row in rows]

    async def set_vectors(self, fact_id: str, vectors: tuple[float, ...], *, expected_seq: int) -> bool:
        """Fill a memory's vectors only if it still has `expected_seq` and is unchanged since read (etag). Not a change:
        no change record. A lost race is skipped, never retried."""
        from azure.core import MatchConditions  # noqa: PLC0415
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError  # noqa: PLC0415

        doc = await self._read_raw(fact_id)
        if doc is None or "doc_type" in doc or doc.get("seq", 0) != expected_seq:
            return False
        body = {key: value for key, value in doc.items() if key not in SYSTEM_KEYS}
        body["vector_embeddings"] = list(vectors)
        try:
            await asyncio.to_thread(lambda: self._container.replace_item(
                item=fact_id, body=body, etag=doc["_etag"], match_condition=MatchConditions.IfNotModified))
        except CosmosAccessConditionFailedError:
            return False
        return True

    async def upsert_entity(self, entity: EntityRecord) -> None:
        doc = self._stamp(entity_to_state(entity))
        await asyncio.to_thread(lambda: self._container.upsert_item(doc))

    async def read_entity(self, entity_id: str) -> EntityRecord | None:
        doc = await self._read_raw(entity_id)
        return entity_from_state(doc) if doc is not None and doc.get("doc_type") == "entity" else None

    async def read_entities(self, ids: Sequence[str]) -> list[EntityRecord]:
        """The entities with these ids, in one query, sorted by id; ids with no entity are skipped. At most
        READ_ENTITIES_MAX_IDS distinct ids."""
        distinct = check_entity_ids(ids)
        if not distinct:
            return []
        return await self._entities([CLAUSE_ENTITY_IDS], [{"name": "@ids", "value": distinct}])

    async def list_entities(self, *, type: str | None = None) -> list[EntityRecord]:
        if type is None:
            return await self._entities([], [])
        return await self._entities([CLAUSE_ENTITY_TYPE], [{"name": "@type", "value": type}])

    async def find_by_alias(self, type: str, alias: str) -> list[EntityRecord]:
        return await self._entities([CLAUSE_ENTITY_TYPE, CLAUSE_ALIAS],
                                    [{"name": "@type", "value": type}, {"name": "@alias", "value": alias_key(alias)}])

    async def find_by_identifier(self, kind: str, value: str) -> list[EntityRecord]:
        return await self._entities([CLAUSE_IDENTIFIER],
                                    [{"name": "@identifier", "value": identifier_key(kind, value)}])

    # --- helpers ---

    async def _entities(self, filters: list[str], params: list[dict[str, Any]]) -> list[EntityRecord]:
        query = "SELECT * FROM c WHERE " + " AND ".join([CLAUSE_ENTITIES, *filters])
        rows = await asyncio.to_thread(
            lambda: list(self._container.query_items(query=query, parameters=params or None, partition_key=self._pk))
        )
        return sorted((entity_from_state(row) for row in rows), key=lambda e: e.entity_id)

    def _stamp(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {**doc, "family_id": self._family_id, "user_id": self._scope, "category": "FACTS"}

    async def _read_raw(self, item_id: str) -> dict[str, Any] | None:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError  # noqa: PLC0415

        try:
            return await asyncio.to_thread(lambda: self._container.read_item(item=item_id, partition_key=self._pk))
        except CosmosResourceNotFoundError:
            return None

    def _memory_filters(self, *, include_expired: bool, include_future: bool) -> tuple[list[str], list[dict[str, Any]]]:
        filters = [CLAUSE_MEMORY, CLAUSE_NOT_DELETED, CLAUSE_NOT_FORGOTTEN, CLAUSE_NOT_PURGED]
        if not include_expired:
            filters.append(CLAUSE_NOT_EXPIRED)
        if not include_future:
            filters.append(CLAUSE_STARTED)
        params = [] if include_expired and include_future else [
            {"name": "@now", "value": read_clock(self._clock).isoformat()}]
        return filters, params

    def _to_document(self, record: FactRecord) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "family_id": self._family_id,
            "user_id": self._scope,
            "category": "FACTS",
            "is_shared": self._scope == "shared",
            **fact_state(record),
        }
        if record.vector_embeddings:
            doc["vector_embeddings"] = list(record.vector_embeddings)
        return doc

    def _from_document(self, doc: dict[str, Any]) -> FactRecord:
        return record_from_state(doc, now=read_clock(self._clock))
