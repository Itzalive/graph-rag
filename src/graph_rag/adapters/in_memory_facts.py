from __future__ import annotations

from datetime import UTC, datetime

from graph_rag.facts import FactRecord, FactSearchHit, FactSearchRequest, cosine_distance


class InMemoryFactBackend:
    """Deterministic in-memory FactBackend for testing. Uses cosine distance for vector search."""

    def __init__(self) -> None:
        self._facts: dict[str, FactRecord] = {}

    async def upsert_fact(self, record: FactRecord) -> None:
        self._facts[record.fact_id] = record

    async def read_fact(self, fact_id: str) -> FactRecord | None:
        return self._facts.get(fact_id)

    async def list_facts(self, *, include_expired: bool = False) -> list[FactRecord]:
        now = datetime.now(UTC)
        result = []
        for record in self._facts.values():
            if record.deleted:
                continue
            if not include_expired and record.valid_until is not None and record.valid_until <= now:
                continue
            result.append(record)
        return result

    async def search_facts(self, request: FactSearchRequest) -> list[FactSearchHit]:
        if request.limit > 100:
            raise ValueError("limit must not exceed 100")
        now = datetime.now(UTC)
        exclude = set(request.exclude_ids)
        scored: list[tuple[float, FactRecord]] = []
        for record in self._facts.values():
            if record.deleted:
                continue
            if record.fact_id in exclude:
                continue
            if not request.include_expired and record.valid_until is not None and record.valid_until <= now:
                continue
            if record.vector_embeddings is None:
                continue
            dist = cosine_distance(request.query_vector, record.vector_embeddings)
            scored.append((dist, record))
        scored.sort(key=lambda t: (t[0], t[1].fact_id))
        return [
            FactSearchHit(
                fact_id=r.fact_id,
                body=r.body,
                score=dist,
                created_at=r.created_at,
                updated_at=r.updated_at,
                source=r.source,
                subject=r.subject,
                fact_type=r.fact_type,
                confidence=r.confidence,
                valid_until=r.valid_until,
                extras=r.extras,
            )
            for dist, r in scored[: request.limit]
        ]
