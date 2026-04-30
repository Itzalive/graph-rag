from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


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


@dataclass(frozen=True)
class FactSearchRequest:
    query_vector: tuple[float, ...]
    limit: int
    exclude_ids: tuple[str, ...] = ()
    include_expired: bool = False

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


@runtime_checkable
class FactBackend(Protocol):
    async def upsert_fact(self, record: FactRecord) -> None: ...
    async def read_fact(self, fact_id: str) -> FactRecord | None: ...
    async def list_facts(self, *, include_expired: bool = False) -> list[FactRecord]: ...
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
