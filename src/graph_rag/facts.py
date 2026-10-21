from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Holder:
    """Whose statement a memory is: a kind (for example a person or the system itself) and, optionally, who."""
    kind: str
    ref: str | None = None


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    body: str
    created_at: datetime
    updated_at: datetime
    source: str | None = None
    subject: str | None = None
    fact_type: str | None = None
    confidence: float | None = None
    valid_until: datetime | None = None
    deleted: bool = False
    deleted_at: datetime | None = None
    vector_embeddings: tuple[float, ...] | None = None
    extras: dict[str, Any] = field(default_factory=dict, hash=False)
    # Recorded-memory fields (0.3.0). All optional: documents written before them read as kind None, seq 0.
    kind: str | None = None
    holder: Holder | None = None
    lifespan: str | None = None
    valid_from: datetime | None = None
    validity_basis: str | None = None
    forgotten_at: datetime | None = None
    purged_at: datetime | None = None
    learned_at: datetime | None = None
    last_changed_at: datetime | None = None
    last_confirmed_at: datetime | None = None
    entities: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    seq: int = 0


@dataclass(frozen=True)
class FactSearchRequest:
    query_vector: tuple[float, ...]
    limit: int
    exclude_ids: tuple[str, ...] = ()
    include_expired: bool = False
    include_future: bool = False

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("limit must be positive")


@dataclass(frozen=True)
class FactSearchHit:
    fact_id: str
    body: str
    score: float  # Cosine distance from vector search; lower is better.
    created_at: datetime | None = None
    updated_at: datetime | None = None
    source: str | None = None
    subject: str | None = None
    fact_type: str | None = None
    confidence: float | None = None
    valid_until: datetime | None = None
    extras: dict[str, Any] = field(default_factory=dict, hash=False)
    kind: str | None = None
    holder: Holder | None = None
    lifespan: str | None = None
    valid_from: datetime | None = None
    validity_basis: str | None = None
    learned_at: datetime | None = None
    last_changed_at: datetime | None = None
    last_confirmed_at: datetime | None = None
    entities: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    seq: int = 0


@runtime_checkable
class FactBackend(Protocol):
    async def upsert_fact(self, record: FactRecord) -> None: ...
    async def read_fact(self, fact_id: str) -> FactRecord | None: ...
    async def list_facts(self, *, include_expired: bool = False, include_future: bool = False) -> list[FactRecord]: ...
    async def search_facts(self, request: FactSearchRequest) -> list[FactSearchHit]: ...


def cosine_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Returns cosine distance in [0, 2]. Lower = more similar."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimensions")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)


SYSTEM_KEYS = frozenset({"_rid", "_self", "_etag", "_attachments", "_ts"})
PARTITION_KEYS = frozenset({"family_id", "user_id", "category", "is_shared"})
_CORE_KEYS = frozenset({
    "id", "fact", "created_at", "last_updated_at", "source", "subject", "fact_type", "confidence", "valid_until",
    "deleted", "deleted_at", "vector_embeddings",
})
RECORDED_KEYS = frozenset({
    "kind", "holder", "lifespan", "valid_from", "validity_basis", "forgotten_at", "purged_at", "learned_at",
    "last_changed_at", "last_confirmed_at", "entities", "topics", "seq",
})
# Keys a stored memory document holds that are not extras.
NON_EXTRA_KEYS = SYSTEM_KEYS | PARTITION_KEYS | _CORE_KEYS | RECORDED_KEYS


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _holder_state(holder: Holder | None) -> dict[str, str] | None:
    if holder is None:
        return None
    return {"kind": holder.kind} if holder.ref is None else {"kind": holder.kind, "ref": holder.ref}


def _holder(value: dict[str, Any] | None) -> Holder | None:
    return None if not value else Holder(value["kind"], value.get("ref"))


def fact_state(record: FactRecord) -> dict[str, Any]:
    """A memory's stored fields, without partition fields or vectors: the memory document's body and a change record's
    `after`. Unset fields are left out, so a record that uses no recorded field gives exactly the pre-0.3.0 document. A
    typed field that is set wins over an extras key of the same name; an unset one leaves the extras value in place."""
    core = {
        "id": record.fact_id,
        "fact": record.body,
        "created_at": record.created_at.isoformat(),
        "last_updated_at": record.updated_at.isoformat(),
        "source": record.source,
        "subject": record.subject,
        "fact_type": record.fact_type,
        "confidence": record.confidence,
        "valid_until": _iso(record.valid_until),
        "deleted": record.deleted,
        "deleted_at": _iso(record.deleted_at),
    }
    recorded = {
        "kind": record.kind,
        "holder": _holder_state(record.holder),
        "lifespan": record.lifespan,
        "valid_from": _iso(record.valid_from),
        "validity_basis": record.validity_basis,
        "forgotten_at": _iso(record.forgotten_at),
        "purged_at": _iso(record.purged_at),
        "learned_at": _iso(record.learned_at),
        "last_changed_at": _iso(record.last_changed_at),
        "last_confirmed_at": _iso(record.last_confirmed_at),
        "entities": list(record.entities) or None,
        "topics": list(record.topics) or None,
        "seq": record.seq or None,
    }
    state = {**core, **record.extras, **{key: value for key, value in recorded.items() if value is not None}}
    return {key: value for key, value in state.items() if value is not None}


def record_from_state(doc: dict[str, Any], *, now: datetime) -> FactRecord:
    """The record a stored memory document holds; `now` stands in for missing creation times."""
    return FactRecord(
        fact_id=doc["id"],
        body=doc["fact"],
        created_at=_dt(doc.get("created_at")) or now,
        updated_at=_dt(doc.get("last_updated_at")) or _dt(doc.get("created_at")) or now,
        source=doc.get("source"),
        subject=doc.get("subject"),
        fact_type=doc.get("fact_type"),
        confidence=doc.get("confidence"),
        valid_until=_dt(doc.get("valid_until")),
        deleted=bool(doc.get("deleted", False)),
        deleted_at=_dt(doc.get("deleted_at")),
        vector_embeddings=tuple(doc["vector_embeddings"]) if doc.get("vector_embeddings") else None,
        extras={key: value for key, value in doc.items() if key not in NON_EXTRA_KEYS},
        kind=doc.get("kind"),
        holder=_holder(doc.get("holder")),
        lifespan=doc.get("lifespan"),
        valid_from=_dt(doc.get("valid_from")),
        validity_basis=doc.get("validity_basis"),
        forgotten_at=_dt(doc.get("forgotten_at")),
        purged_at=_dt(doc.get("purged_at")),
        learned_at=_dt(doc.get("learned_at")),
        last_changed_at=_dt(doc.get("last_changed_at")),
        last_confirmed_at=_dt(doc.get("last_confirmed_at")),
        entities=tuple(doc.get("entities") or ()),
        topics=tuple(doc.get("topics") or ()),
        seq=int(doc.get("seq", 0)),
    )


def hit_from_state(doc: dict[str, Any], score: float) -> FactSearchHit:
    """A search hit from a stored document's selected fields."""
    return FactSearchHit(
        fact_id=doc["id"],
        body=doc["fact"],
        score=score,
        created_at=_dt(doc.get("created_at")),
        updated_at=_dt(doc.get("last_updated_at")),
        source=doc.get("source"),
        subject=doc.get("subject"),
        fact_type=doc.get("fact_type"),
        confidence=doc.get("confidence"),
        valid_until=_dt(doc.get("valid_until")),
        extras={key: value for key, value in doc.items() if key not in NON_EXTRA_KEYS and key != "score"},
        kind=doc.get("kind"),
        holder=_holder(doc.get("holder")),
        lifespan=doc.get("lifespan"),
        valid_from=_dt(doc.get("valid_from")),
        validity_basis=doc.get("validity_basis"),
        learned_at=_dt(doc.get("learned_at")),
        last_changed_at=_dt(doc.get("last_changed_at")),
        last_confirmed_at=_dt(doc.get("last_confirmed_at")),
        entities=tuple(doc.get("entities") or ()),
        topics=tuple(doc.get("topics") or ()),
        seq=int(doc.get("seq", 0)),
    )


def hit_from_record(record: FactRecord, score: float) -> FactSearchHit:
    """A search hit for a record held in memory."""
    return FactSearchHit(
        fact_id=record.fact_id, body=record.body, score=score, created_at=record.created_at,
        updated_at=record.updated_at, source=record.source, subject=record.subject, fact_type=record.fact_type,
        confidence=record.confidence, valid_until=record.valid_until, extras=record.extras, kind=record.kind,
        holder=record.holder, lifespan=record.lifespan, valid_from=record.valid_from,
        validity_basis=record.validity_basis, learned_at=record.learned_at, last_changed_at=record.last_changed_at,
        last_confirmed_at=record.last_confirmed_at, entities=record.entities, topics=record.topics, seq=record.seq,
    )
