from __future__ import annotations

import asyncio
import threading
from typing import Any

from graph_rag.clock import Clock, read_clock, utc_now
from graph_rag.facts import (
    SYSTEM_KEYS, FactRecord, FactSearchHit, FactSearchRequest, fact_state, hit_from_state, record_from_state,
)


# Every fact field a search hit can use, including the extras seen in production facts (claude-assistant-yh9p) and the
# recorded-memory fields with Kerwen's own extras (claude-assistant-xz59.9 §3).
_SEARCH_FIELDS = (
    "id", "fact", "created_at", "last_updated_at", "source", "subject", "fact_type", "confidence", "valid_until",
    "family_id", "user_id", "category", "is_shared", "deleted", "deleted_at",
    "extracted_at", "history", "metadata", "updated_at", "user",
    "kind", "holder", "lifespan", "valid_from", "validity_basis", "forgotten_at", "purged_at", "learned_at",
    "last_changed_at", "last_confirmed_at", "entities", "topics", "seq",
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
        from azure.cosmos.exceptions import CosmosResourceNotFoundError  # noqa: PLC0415

        try:
            doc = await asyncio.to_thread(
                lambda: self._container.read_item(
                    item=fact_id,
                    partition_key=self._pk,
                )
            )
            return None if "doc_type" in doc else self._from_document(doc)
        except CosmosResourceNotFoundError:
            return None

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

    # --- helpers ---

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
