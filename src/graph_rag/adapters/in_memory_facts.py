from __future__ import annotations

from graph_rag.clock import Clock, read_clock, utc_now
from graph_rag.facts import FactRecord, FactSearchHit, FactSearchRequest, cosine_distance, hit_from_record, is_listed


class InMemoryFactBackend:
    """Deterministic in-memory FactBackend for testing. Uses cosine distance for vector search."""

    def __init__(self, *, clock: Clock = utc_now) -> None:
        self._facts: dict[str, FactRecord] = {}
        self._clock = clock

    async def upsert_fact(self, record: FactRecord) -> None:
        self._facts[record.fact_id] = record

    async def read_fact(self, fact_id: str) -> FactRecord | None:
        return self._facts.get(fact_id)

    async def list_facts(self, *, include_expired: bool = False, include_future: bool = False) -> list[FactRecord]:
        now = read_clock(self._clock)
        return [r for r in self._facts.values()
                if is_listed(r, now, include_expired=include_expired, include_future=include_future)]

    async def search_facts(self, request: FactSearchRequest) -> list[FactSearchHit]:
        if request.limit > 100:
            raise ValueError("limit must not exceed 100")
        now = read_clock(self._clock)
        exclude = set(request.exclude_ids)
        scored: list[tuple[float, FactRecord]] = []
        for record in self._facts.values():
            if record.fact_id in exclude:
                continue
            if not is_listed(record, now, include_expired=request.include_expired,
                             include_future=request.include_future):
                continue
            if record.vector_embeddings is None:
                continue
            dist = cosine_distance(request.query_vector, record.vector_embeddings)
            scored.append((dist, record))
        scored.sort(key=lambda t: (t[0], t[1].fact_id))
        return [hit_from_record(r, dist) for dist, r in scored[: request.limit]]
