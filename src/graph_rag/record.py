"""The change record (claude-assistant-xz59.9 §2): every change to a memory is an append-only change record holding
its source, time, reason and the memory's full state after the change, written in the same transaction as the memory."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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


class StaleWriteError(Exception):
    """Another write took this change's seq, or created an entity this write creates: re-read and apply once more."""


class AlreadyAppliedError(Exception):
    """This request (its op_id) was applied before: treat as success with nothing written."""


class WriteCheckError(ValueError):
    """A write that breaks a write check (W1–W4); `code` names the rule. Nothing is written."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def change_id(fact_id: str, seq: int) -> str:
    return f"{fact_id}:{seq}"


def op_marker_id(op_id: str) -> str:
    return f"op-{op_id}"


def check_history_limit(limit: int) -> None:
    if not 1 <= limit <= HISTORY_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {HISTORY_MAX_LIMIT}")


# Cosmos holds at most 100 operations in one transactional batch; a write is the marker, its changes, its new
# entities and the memory.
MAX_BATCH_OPERATIONS = 100
# Characters Cosmos refuses in a document id: the fact id and op id become parts of ids.
_ID_FORBIDDEN = frozenset("/\\?#")


def check_write(record: FactRecord, changes: Sequence[FactChange], entities: Sequence[EntityRecord] = ()) -> None:
    """The library's write checks, run before anything is sent. Raises ValueError for a malformed write and
    WriteCheckError for a memory that breaks W1–W4."""
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
    if len({entity.entity_id for entity in entities}) != len(entities):
        raise ValueError("entity ids must be distinct")
    if 2 + len(changes) + len(entities) > MAX_BATCH_OPERATIONS:
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


@runtime_checkable
class RecordedFactBackend(FactBackend, Protocol):
    async def write_change(self, record: FactRecord, changes: Sequence[FactChange],
                           entities: Sequence[EntityRecord] = ()) -> None: ...
    async def history(self, fact_id: str, *, after_seq: int = -1,
                      limit: int = HISTORY_DEFAULT_LIMIT) -> list[FactChange]: ...
    async def set_vectors(self, fact_id: str, vectors: tuple[float, ...], *, expected_seq: int) -> bool: ...
    async def upsert_entity(self, entity: EntityRecord) -> None: ...
    async def read_entity(self, entity_id: str) -> EntityRecord | None: ...
    async def list_entities(self, *, type: str | None = None) -> list[EntityRecord]: ...
    async def find_by_alias(self, type: str, alias: str) -> list[EntityRecord]: ...
    async def find_by_identifier(self, kind: str, value: str) -> list[EntityRecord]: ...
