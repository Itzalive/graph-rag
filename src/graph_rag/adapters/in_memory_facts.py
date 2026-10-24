from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import replace

from graph_rag.clock import Clock, read_clock, utc_now
from graph_rag.entities import EntityRecord, alias_key, alias_keys, identifier_key, identifier_keys
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
        self._entities: dict[str, EntityRecord] = {}
        self._entity_versions = itertools.count(1)
        self._clock = clock

    def _put_entity(self, entity: EntityRecord) -> None:
        """Store an entity under a new version, as Cosmos gives a written document a new _etag."""
        self._entities[entity.entity_id] = replace(entity, version=str(next(self._entity_versions)))

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

    async def write_change(self, record: FactRecord, changes: Sequence[FactChange],
                           entities: Sequence[EntityRecord] = ()) -> None:
        check_write(record, changes, entities)
        op_id = changes[-1].op_id
        if op_id in self._ops:
            raise AlreadyAppliedError(op_id)
        stored = [*changes[:-1], replace(changes[-1], after=fact_state(record))]
        if any(change_id(c.fact_id, c.seq) in self._changes for c in stored) or any(
                e.entity_id in self._entities for e in entities):
            raise StaleWriteError(record.fact_id)
        self._ops.add(op_id)
        for change in stored:
            self._changes[change_id(change.fact_id, change.seq)] = change
        for entity in entities:
            self._put_entity(entity)
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

    async def upsert_entity(self, entity: EntityRecord) -> None:
        self._put_entity(entity)

    async def read_entity(self, entity_id: str) -> EntityRecord | None:
        return self._entities.get(entity_id)

    async def list_entities(self, *, type: str | None = None) -> list[EntityRecord]:
        return sorted((e for e in self._entities.values() if type is None or e.type == type), key=lambda e: e.entity_id)

    async def find_by_alias(self, type: str, alias: str) -> list[EntityRecord]:
        return [e for e in await self.list_entities(type=type) if alias_key(alias) in alias_keys(e)]

    async def find_by_identifier(self, kind: str, value: str) -> list[EntityRecord]:
        return [e for e in await self.list_entities() if identifier_key(kind, value) in identifier_keys(e)]
