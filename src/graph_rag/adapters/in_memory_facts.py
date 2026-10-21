from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from graph_rag.clock import Clock, read_clock, utc_now
from graph_rag.facts import (
    FactRecord, FactSearchHit, FactSearchRequest, cosine_distance, fact_state, hit_from_record, is_listed,
)
from graph_rag.record import (
    HISTORY_DEFAULT_LIMIT, AlreadyAppliedError, FactChange, StaleWriteError, change_id, check_history_limit, check_write,
)


class InMemoryFactBackend:
    """Deterministic in-memory RecordedFactBackend for tests and evals: one bucket per instance. Uses cosine distance for
    vector search."""

    def __init__(self, *, clock: Clock = utc_now) -> None:
        self._facts: dict[str, FactRecord] = {}
        self._changes: dict[str, FactChange] = {}
        self._ops: set[str] = set()
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

    async def write_change(self, record: FactRecord, changes: Sequence[FactChange]) -> None:
        check_write(record, changes)
        op_id = changes[-1].op_id
        if op_id in self._ops:
            raise AlreadyAppliedError(op_id)
        stored = [*changes[:-1], replace(changes[-1], after=fact_state(record))]
        if any(change_id(c.fact_id, c.seq) in self._changes for c in stored):
            raise StaleWriteError(record.fact_id)
        self._ops.add(op_id)
        for change in stored:
            self._changes[change_id(change.fact_id, change.seq)] = change
        self._facts[record.fact_id] = record

    async def history(self, fact_id: str, *, after_seq: int = -1,
                      limit: int = HISTORY_DEFAULT_LIMIT) -> list[FactChange]:
        check_history_limit(limit)
        rows = [c for c in self._changes.values() if c.fact_id == fact_id and c.seq > after_seq]
        return sorted(rows, key=lambda c: c.seq)[:limit]

    async def set_vectors(self, fact_id: str, vectors: tuple[float, ...], *, expected_seq: int) -> bool:
        record = self._facts.get(fact_id)
        if record is None or record.seq != expected_seq:
            return False
        self._facts[fact_id] = replace(record, vector_embeddings=tuple(vectors))
        return True
