"""The change record (claude-assistant-xz59.9 §2): every change to a memory is an append-only change record holding
its source, time, reason and the memory's full state after the change, written in the same transaction as the memory."""
from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from graph_rag.entities import EntityRecord
from graph_rag.facts import FactBackend, FactRecord

HISTORY_DEFAULT_LIMIT = 100
HISTORY_MAX_LIMIT = 500


@dataclass(frozen=True)
class ChangeSource:
    """Who or what caused a change: a kind, and optionally the actor, a reference to the source and a display label."""
    kind: str
    actor: str | None = None
    ref: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class FactChange:
    fact_id: str
    seq: int
    kind: str
    at: datetime
    source: ChangeSource
    reason: str
    op_id: str
    # The memory's full state after this change, without vectors. The backend sets it for the last change of a write;
    # an earlier change in the same write (an old document's first entry) carries its own.
    after: dict[str, Any] | None = field(default=None, hash=False)
    approved_by: str | None = None


@dataclass(frozen=True)
class EntityChange:
    """One write of an entity (S2 spec §6): the entity's seq after the write, `created` or `changed`, when, who, and
    the entity's state after it. Stored beside the entity, written in the same batch."""
    entity_id: str
    seq: int
    kind: str
    at: datetime
    source: ChangeSource
    after: dict[str, Any] = field(hash=False)


class StaleWriteError(Exception):
    """Another write took this change's seq, or created an entity this write creates, or an entity `replace_entities`
    updates moved on or is gone: re-read and apply once more."""


class AlreadyAppliedError(Exception):
    """This request (its op_id) was applied before: treat as success with nothing written."""


@dataclass(frozen=True)
class OpMarker:
    """An applied request's marker: the seq its write left, None for a marker written before markers kept it."""
    seq: int | None


class WriteCheckError(ValueError):
    """A write that breaks a write check (W1–W4); `code` names the rule. Nothing is written."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def change_id(fact_id: str, seq: int) -> str:
    return f"{fact_id}:{seq}"


def op_marker_id(op_id: str) -> str:
    return f"op-{op_id}"


def entity_change_id(entity_id: str, seq: int) -> str:
    return f"entity_change:{entity_id}:{seq}"


def check_history_limit(limit: int) -> None:
    if not 1 <= limit <= HISTORY_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {HISTORY_MAX_LIMIT}")


# Cosmos holds at most 100 operations in one transactional batch; a write is the marker, its changes, its new
# entities, its entity updates, a change record for each entity and the memory.
MAX_BATCH_OPERATIONS = 100
# replace_entities writes each update and its change record: two operations each.
MAX_ENTITY_REPLACEMENTS = MAX_BATCH_OPERATIONS // 2
# Characters Cosmos refuses in a document id: the fact id and op id become parts of ids.
_ID_FORBIDDEN = frozenset("/\\?#")


def check_write(record: FactRecord, changes: Sequence[FactChange], entities: Sequence[EntityRecord] = (),
                entity_updates: Sequence[EntityRecord] = ()) -> None:
    """The library's write checks, run before anything is sent. Raises ValueError for a malformed write (a situation without a channel is malformed) and
    WriteCheckError for a memory that breaks W1–W4. An entity update carries the version it read."""
    if not 1 <= len(changes) <= 2:
        raise ValueError("a write carries one or two changes")
    if any(change.fact_id != record.fact_id for change in changes):
        raise ValueError("every change must name the written memory's fact_id")
    first = changes[0].seq
    if [change.seq for change in changes] != list(range(first, first + len(changes))):
        raise ValueError("change seqs must be consecutive")
    if len(changes) == 2 and first != 0:
        raise ValueError("two changes only for an old document's first change, at seq 0")
    if not changes[-1].op_id:
        raise ValueError("the last change must carry an op_id")
    if _ID_FORBIDDEN & set(record.fact_id + changes[-1].op_id):
        raise ValueError("fact_id and op_id must not contain /, \\, ? or #")
    entity_ids = [entity.entity_id for entity in (*entities, *entity_updates)]
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("entity ids must be distinct")
    if any(update.version is None for update in entity_updates):
        raise ValueError("an entity update must carry the version it read")
    if record.situation is not None and not record.situation.channels:
        raise ValueError("a situation must name at least one channel")
    if 2 + len(changes) + 2 * (len(entities) + len(entity_updates)) > MAX_BATCH_OPERATIONS:
        raise ValueError(f"a write may hold at most {MAX_BATCH_OPERATIONS} batch operations")
    if changes[-1].after is not None:
        raise ValueError("the backend sets the last change's after")
    if len(changes) == 2 and changes[0].after is None:
        raise ValueError("an earlier change must carry its own after")
    if record.lifespan == "dated" and record.valid_until is None:
        raise WriteCheckError("W1")
    if record.valid_until is not None and record.lifespan != "dated" and record.validity_basis != "ended":
        raise WriteCheckError("W2")
    if (record.valid_from is not None or record.valid_until is not None) and not record.validity_basis:
        raise WriteCheckError("W3")
    if record.seq != changes[-1].seq:
        raise WriteCheckError("W4")


def check_entity_replacements(updates: Sequence[EntityRecord]) -> None:
    """At most MAX_ENTITY_REPLACEMENTS updates (each takes two batch operations: the replace and its change record),
    distinct ids, each carrying the version it read; otherwise ValueError."""
    if len(updates) > MAX_ENTITY_REPLACEMENTS:
        raise ValueError(f"replace_entities takes at most {MAX_ENTITY_REPLACEMENTS} updates")
    ids = [update.entity_id for update in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("entity ids must be distinct")
    if any(update.version is None for update in updates):
        raise ValueError("an entity update must carry the version it read")


def change_to_state(change: FactChange) -> dict[str, Any]:
    source = {key: value for key, value in vars(change.source).items() if value is not None}
    state = {
        "id": change_id(change.fact_id, change.seq),
        "doc_type": "fact_change",
        "fact_id": change.fact_id,
        "seq": change.seq,
        "kind": change.kind,
        "at": change.at.isoformat(),
        "source": source,
        "reason": change.reason,
        "after": change.after,
        "op_id": change.op_id,
        "approved_by": change.approved_by,
    }
    return {key: value for key, value in state.items() if value is not None}


def change_from_state(doc: dict[str, Any]) -> FactChange:
    return FactChange(
        fact_id=doc["fact_id"],
        seq=doc["seq"],
        kind=doc["kind"],
        at=datetime.fromisoformat(doc["at"]),
        source=ChangeSource(**doc["source"]),
        reason=doc["reason"],
        op_id=doc["op_id"],
        after=doc.get("after"),
        approved_by=doc.get("approved_by"),
    )


def entity_after(entity: EntityRecord) -> dict[str, Any]:
    """An entity's state as its change record keeps it: no version, no lookup keys; `merged_into` only when set."""
    state: dict[str, Any] = {"type": entity.type, "name": entity.name, "handle": entity.handle,
                             "aliases": list(entity.aliases),
                             "identifiers": [{"kind": i.kind, "value": i.value} for i in entity.identifiers],
                             "seq": entity.seq}
    if entity.merged_into is not None:
        state["merged_into"] = entity.merged_into
    return state


def entity_writes(entities: Sequence[EntityRecord], updates: Sequence[EntityRecord], *, source: ChangeSource,
                  at: datetime) -> tuple[list[EntityRecord], list[EntityRecord], list[EntityChange]]:
    """The entities as the backend writes them and their change records (S2 spec §6 The seq): a created entity at
    seq 1 and an update at its seq plus one, whatever seq the caller gave; records in the same order, created first."""
    created = [replace(entity, seq=1) for entity in entities]
    updated = [replace(update, seq=update.seq + 1) for update in updates]
    records = [EntityChange(entity_id=e.entity_id, seq=e.seq, kind=kind, at=at, source=source, after=entity_after(e))
               for kind, written in (("created", created), ("changed", updated)) for e in written]
    return created, updated, records


def entity_change_to_state(change: EntityChange) -> dict[str, Any]:
    return {"id": entity_change_id(change.entity_id, change.seq), "doc_type": "entity_change",
            "entity_id": change.entity_id, "seq": change.seq, "kind": change.kind, "at": change.at.isoformat(),
            "source": {key: value for key, value in vars(change.source).items() if value is not None},
            "after": change.after}


def entity_change_from_state(doc: dict[str, Any]) -> EntityChange:
    return EntityChange(entity_id=doc["entity_id"], seq=doc["seq"], kind=doc["kind"],
                        at=datetime.fromisoformat(doc["at"]), source=ChangeSource(**doc["source"]),
                        after=doc["after"])


@runtime_checkable
class RecordedFactBackend(FactBackend, Protocol):
    async def write_change(self, record: FactRecord, changes: Sequence[FactChange],
                           entities: Sequence[EntityRecord] = (),
                           entity_updates: Sequence[EntityRecord] = ()) -> None: ...
    async def op_applied(self, op_id: str) -> bool: ...
    async def op_marker(self, op_id: str) -> OpMarker | None: ...
    async def history(self, fact_id: str, *, after_seq: int = -1,
                      limit: int = HISTORY_DEFAULT_LIMIT) -> list[FactChange]: ...
    async def set_vectors(self, fact_id: str, vectors: tuple[float, ...], *, expected_seq: int) -> bool: ...
    async def upsert_entity(self, entity: EntityRecord) -> None: ...
    async def read_entity(self, entity_id: str) -> EntityRecord | None: ...
    async def read_entities(self, ids: Sequence[str]) -> list[EntityRecord]: ...
    async def list_entities(self, *, type: str | None = None) -> list[EntityRecord]: ...
    async def find_by_alias(self, type: str, alias: str) -> list[EntityRecord]: ...
    async def find_by_identifier(self, kind: str, value: str) -> list[EntityRecord]: ...
    async def situational(self, channel: str, types: Collection[str], entity_ids: Collection[str], *,
                          unsituated_kinds: Collection[str] = (), limit: int = 100) -> list[FactRecord]: ...
    async def find_by_aliases(self, keys: Iterable[str], *, limit: int = 50) -> list[EntityRecord]: ...
    async def replace_entities(self, updates: Sequence[EntityRecord], *, source: ChangeSource,
                               at: datetime) -> None: ...
    async def entity_history(self, entity_id: str, *, after_seq: int = -1,
                             limit: int = HISTORY_DEFAULT_LIMIT) -> list[EntityChange]: ...
