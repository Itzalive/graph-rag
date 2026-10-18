from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from typing import Any

from graph_rag.clock import Clock, read_clock, utc_now
from graph_rag.facts import FactRecord, FactSearchHit, FactSearchRequest


# Every fact field a search hit can use, including the extras seen in production facts (claude-assistant-yh9p).
_SEARCH_FIELDS = (
    "id", "fact", "created_at", "last_updated_at", "source", "subject", "fact_type", "confidence", "valid_until",
    "family_id", "user_id", "category", "is_shared", "deleted", "deleted_at",
    "extracted_at", "history", "metadata", "updated_at", "user",
)
_SEARCH_SELECT = ", ".join(f"c.{field}" for field in _SEARCH_FIELDS)


class CosmosFactBackend:
    """Cosmos DB-backed FactBackend.

    Constructor injection: pass a pre-built ``ContainerProxy`` or supply
    ``cosmos_url``, ``credential``, and ``database_name`` to build one lazily.
    The container name is always ``"memory"`` (the shared assistant container).
    """

    def __init__(
        self,
        container=None,
        *,
        cosmos_url: str | None = None,
        credential=None,
        database_name: str | None = None,
        family_id: str,
        clock: Clock = utc_now,
    ) -> None:
        if container is None and (cosmos_url is None or credential is None or database_name is None):
            raise ValueError(
                "Provide either a ContainerProxy or all of: cosmos_url, credential, database_name"
            )
        # A URL-built client is made on first use, inside the worker thread the operation runs in: the CosmosClient
        # constructor fetches the database account over the network (claude-assistant-5si8).
        self._container_client = container
        self._cosmos_url = cosmos_url
        self._credential = credential
        self._database_name = database_name
        self._build_lock = threading.Lock()
        self._family_id = family_id
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
                    partition_key=[self._family_id, "shared", "FACTS"],
                )
            )
            return self._from_document(doc)
        except CosmosResourceNotFoundError:
            return None

    async def list_facts(self, *, include_expired: bool = False) -> list[FactRecord]:
        filters = ["(NOT IS_DEFINED(c.deleted) OR c.deleted != true)"]
        params: list[dict[str, Any]] = []
        if not include_expired:
            filters.append(
                "(NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until) OR c.valid_until > @now)"
            )
            params.append({"name": "@now", "value": read_clock(self._clock).isoformat()})
        query = "SELECT * FROM c WHERE " + " AND ".join(filters)
        results = await asyncio.to_thread(
            lambda: list(
                self._container.query_items(
                    query=query,
                    parameters=params or None,
                    partition_key=[self._family_id, "shared", "FACTS"],
                )
            )
        )
        return [self._from_document(doc) for doc in results]

    async def search_facts(self, request: FactSearchRequest) -> list[FactSearchHit]:
        if request.limit > 100:
            raise ValueError("limit must not exceed 100")
        filters = ["(NOT IS_DEFINED(c.deleted) OR c.deleted != true)"]
        params: list[dict[str, Any]] = [{"name": "@qv", "value": list(request.query_vector)}]
        if request.exclude_ids:
            filters.append("ARRAY_CONTAINS(@exclude_ids, c.id) = false")
            params.append({"name": "@exclude_ids", "value": list(request.exclude_ids)})
        if not request.include_expired:
            filters.append(
                "(NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until) OR c.valid_until > @now)"
            )
            params.append({"name": "@now", "value": read_clock(self._clock).isoformat()})
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
                    partition_key=[self._family_id, "shared", "FACTS"],
                )
            )
        )
        results = raw_results
        known_keys = {
            "id", "fact", "score", "created_at", "last_updated_at", "source",
            "subject", "fact_type", "confidence", "valid_until", "family_id",
            "user_id", "category", "is_shared", "deleted", "deleted_at",
            "vector_embeddings", "_rid", "_self", "_etag", "_attachments", "_ts",
        }
        return [
            FactSearchHit(
                fact_id=doc["id"],
                body=doc["fact"],
                score=doc.get("score", float("inf")),
                created_at=self._parse_datetime(doc.get("created_at")),
                updated_at=self._parse_datetime(doc.get("last_updated_at")),
                source=doc.get("source"),
                subject=doc.get("subject"),
                fact_type=doc.get("fact_type"),
                confidence=doc.get("confidence"),
                valid_until=self._parse_datetime(doc.get("valid_until")),
                extras={k: v for k, v in doc.items() if k not in known_keys},
            )
            for doc in results
        ]

    # --- helpers ---

    def _to_document(self, record: FactRecord) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "id": record.fact_id,
            "family_id": self._family_id,
            "user_id": "shared",
            "category": "FACTS",
            "is_shared": True,
            "fact": record.body,
            "created_at": record.created_at.isoformat(),
            "last_updated_at": record.updated_at.isoformat(),
            "source": record.source,
            "subject": record.subject,
            "fact_type": record.fact_type,
            "confidence": record.confidence,
            "deleted": record.deleted,
            "deleted_at": record.deleted_at.isoformat() if record.deleted_at else None,
            "vector_embeddings": list(record.vector_embeddings) if record.vector_embeddings else None,
            **record.extras,
        }
        if record.valid_until is not None:
            doc["valid_until"] = record.valid_until.isoformat()
        return {k: v for k, v in doc.items() if v is not None}

    def _from_document(self, doc: dict[str, Any]) -> FactRecord:
        known_keys = {
            "id", "family_id", "user_id", "category", "is_shared",
            "fact", "created_at", "last_updated_at", "source", "subject",
            "fact_type", "confidence", "valid_until", "deleted", "deleted_at",
            "vector_embeddings", "_rid", "_self", "_etag", "_attachments", "_ts",
        }
        extras = {k: v for k, v in doc.items() if k not in known_keys}
        return FactRecord(
            fact_id=doc["id"],
            body=doc["fact"],
            created_at=self._parse_datetime(doc.get("created_at")) or read_clock(self._clock),
            updated_at=(
                self._parse_datetime(doc.get("last_updated_at"))
                or self._parse_datetime(doc.get("created_at"))
                or read_clock(self._clock)
            ),
            source=doc.get("source"),
            subject=doc.get("subject"),
            fact_type=doc.get("fact_type"),
            confidence=doc.get("confidence"),
            valid_until=self._parse_datetime(doc.get("valid_until")),
            deleted=bool(doc.get("deleted", False)),
            deleted_at=self._parse_datetime(doc.get("deleted_at")),
            vector_embeddings=tuple(doc["vector_embeddings"]) if doc.get("vector_embeddings") else None,
            extras=extras,
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if value is None:
            return None
        return datetime.fromisoformat(value)
