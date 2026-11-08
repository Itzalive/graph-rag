from __future__ import annotations

import itertools
from collections.abc import Collection, Iterable, Sequence
from dataclasses import replace
from datetime import datetime

from graph_rag.clock import Clock, read_clock, utc_now
from graph_rag.entities import (
    ALIASES_MAX_READ, EntityRecord, alias_key, alias_keys, alias_lookup_keys, check_alias_limit,
    check_entity_ids, identifier_key, identifier_keys, rank_alias_matches,
)
from graph_rag.events import NO_EVENTS, EventSink
from graph_rag.facts import (
    FactRecord, FactSearchHit, FactSearchRequest, check_situational, cosine_distance, fact_state, hit_from_record,
    is_listed, situation_applies, situational_key,
)
from graph_rag.record import (
    HISTORY_DEFAULT_LIMIT, AlreadyAppliedError, ChangeSource, EntityChange, FactChange, OpMarker, StaleWriteError,
    change_id, check_entity_replacements, check_history_limit, check_write, entity_change_id, entity_writes,
)


class InMemoryFactBackend:
    """Deterministic in-memory RecordedFactBackend for tests and evals: one bucket per instance. Uses cosine distance for
    vector search. Limits reached by a read are counted on the event sink (default: none)."""

    def __init__(self, *, clock: Clock = utc_now, events: EventSink = NO_EVENTS) -> None:
        self._facts: dict[str, FactRecord] = {}
        self._changes: dict[str, FactChange] = {}
        self._ops: dict[str, int] = {}
        self._entities: dict[str, EntityRecord] = {}
        self._entity_changes: dict[str, EntityChange] = {}
        self._entity_versions = itertools.count(1)
        self._clock = clock
        self._events = events

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
        excluded_versions = set(request.exclude_versions)
        scored: list[tuple[float, FactRecord]] = []
        for record in self._facts.values():
            if record.fact_id in exclude or (record.fact_id, record.seq) in excluded_versions:
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

    def _cut(self, rows: list[FactRecord], query: str, limit: int) -> list[FactRecord]:
        if len(rows) > limit:
            self._events.count("situational.limit_reached", query=query, limit=limit)
            rows = rows[:limit]
        return [replace(row, vector_embeddings=None) for row in rows]

    async def situational(self, channel: str, types: Collection[str], entity_ids: Collection[str], *,
                          unsituated_kinds: Collection[str] = (), limit: int = 100) -> list[FactRecord]:
        """Current memories whose situation applies, then (only when kinds are passed) current memories of those kinds
        with no situation; each read in the situational order and cut to `limit`. A result is for reading only: it
        carries no vectors, so a caller that writes a memory found here reads it with `read_fact` first."""
        type_list, id_list, kind_list = check_situational(channel, types, entity_ids, limit, unsituated_kinds)
        now = read_clock(self._clock)
        current = [r for r in self._facts.values() if is_listed(r, now)]
        situated = sorted((r for r in current if r.situation is not None
                           and situation_applies(r.situation, channel, type_list, id_list)), key=situational_key)
        kinds = set(kind_list)
        unsituated = sorted((r for r in current if kinds and r.situation is None and r.kind in kinds),
                            key=situational_key)
        return [*self._cut(situated, "situated", limit), *self._cut(unsituated, "unsituated", limit)]

    async def write_change(self, record: FactRecord, changes: Sequence[FactChange],
                           entities: Sequence[EntityRecord] = (),
                           entity_updates: Sequence[EntityRecord] = ()) -> None:
        check_write(record, changes, entities, entity_updates)
        created, updated, records = entity_writes(entities, entity_updates, source=changes[-1].source,
                                                  at=changes[-1].at)
        op_id = changes[-1].op_id
        if op_id in self._ops:
            raise AlreadyAppliedError(op_id)
        stored = [*changes[:-1], replace(changes[-1], after=fact_state(record))]
        if any(change_id(c.fact_id, c.seq) in self._changes for c in stored) or any(
                e.entity_id in self._entities for e in entities) or any(
                self._changed_since_read(update) for update in entity_updates) or any(
                entity_change_id(r.entity_id, r.seq) in self._entity_changes for r in records):
            raise StaleWriteError(record.fact_id)
        self._ops[op_id] = record.seq
        for change in stored:
            self._changes[change_id(change.fact_id, change.seq)] = change
        for entity in (*created, *updated):
            self._put_entity(entity)
        for entity_record in records:
            self._entity_changes[entity_change_id(entity_record.entity_id, entity_record.seq)] = entity_record
        self._facts[record.fact_id] = record

    async def op_applied(self, op_id: str) -> bool:
        return op_id in self._ops

    async def op_marker(self, op_id: str) -> OpMarker | None:
        return OpMarker(self._ops[op_id]) if op_id in self._ops else None

    def _changed_since_read(self, update: EntityRecord) -> bool:
        current = self._entities.get(update.entity_id)
        return current is None or current.version != update.version

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

    async def replace_entities(self, updates: Sequence[EntityRecord], *, source: ChangeSource,
                               at: datetime) -> None:
        """Replace these entities, each at its seq plus one, and store each one's change record; all or none. An
        entity moved on or gone since read, or a change record id already stored, raises StaleWriteError naming that
        entity before anything is written."""
        check_entity_replacements(updates)
        _, written, records = entity_writes((), updates, source=source, at=at)
        for update in updates:
            if self._changed_since_read(update):
                raise StaleWriteError(update.entity_id)
        for record in records:
            if entity_change_id(record.entity_id, record.seq) in self._entity_changes:
                raise StaleWriteError(record.entity_id)
        for entity in written:
            self._put_entity(entity)
        for record in records:
            self._entity_changes[entity_change_id(record.entity_id, record.seq)] = record

    async def entity_history(self, entity_id: str, *, after_seq: int = -1,
                             limit: int = HISTORY_DEFAULT_LIMIT) -> list[EntityChange]:
        """This entity's change records with a seq above `after_seq`, in seq order, at most `limit`."""
        check_history_limit(limit)
        rows = [c for c in self._entity_changes.values() if c.entity_id == entity_id and c.seq > after_seq]
        return sorted(rows, key=lambda c: c.seq)[:limit]

    async def read_entity(self, entity_id: str) -> EntityRecord | None:
        return self._entities.get(entity_id)

    async def read_entities(self, ids: Sequence[str]) -> list[EntityRecord]:
        return [self._entities[i] for i in check_entity_ids(ids) if i in self._entities]

    async def list_entities(self, *, type: str | None = None) -> list[EntityRecord]:
        return sorted((e for e in self._entities.values() if type is None or e.type == type), key=lambda e: e.entity_id)

    async def find_by_alias(self, type: str, alias: str) -> list[EntityRecord]:
        return [e for e in await self.list_entities(type=type) if alias_key(alias) in alias_keys(e)]

    async def find_by_aliases(self, keys: Iterable[str], *, limit: int = 50) -> list[EntityRecord]:
        check_alias_limit(limit)
        used, given = alias_lookup_keys(keys)
        if given > len(used):
            self._events.count("aliases.keys_capped", given=given, used=len(used))
        if not used:
            return []
        wanted = set(used)
        matches = [e for e in await self.list_entities() if wanted.intersection(alias_keys(e))]
        if len(matches) > ALIASES_MAX_READ:
            matches = matches[:ALIASES_MAX_READ]
            self._events.count("aliases.read_capped")
        ranked = rank_alias_matches(matches, used)
        if len(ranked) > limit:
            self._events.count("aliases.limit_reached", limit=limit)
        return ranked[:limit]

    async def find_by_identifier(self, kind: str, value: str) -> list[EntityRecord]:
        return [e for e in await self.list_entities() if identifier_key(kind, value) in identifier_keys(e)]
