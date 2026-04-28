from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    body: str
    subject: str
    fact_type: str
    confidence: float
    created_at: datetime
    updated_at: datetime
    valid_until: datetime | None
    source: str
    deleted: bool


@dataclass(frozen=True)
class SearchQuery:
    text: str
    subject: str | None
    limit: int


@dataclass(frozen=True)
class MarkdownDocument:
    path: str
    content: str


class FactStore(Protocol):
    async def create_fact(self, record: FactRecord) -> FactRecord:
        pass

    async def read_fact(self, fact_id: str) -> FactRecord | None:
        pass

    async def search_facts(self, query: SearchQuery) -> list[FactRecord]:
        pass

    async def export_markdown(self) -> list[MarkdownDocument]:
        pass
