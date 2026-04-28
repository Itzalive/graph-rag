from __future__ import annotations

from graph_rag.contracts import FactRecord, MarkdownDocument, SearchQuery
from graph_rag.exceptions import FactConflictError


class InMemoryFactStore:
    def __init__(self) -> None:
        self._items: dict[str, FactRecord] = {}

    async def create_fact(self, record: FactRecord) -> FactRecord:
        if not 0.0 <= record.confidence <= 1.0:
            raise ValueError("confidence")
        existing = self._items.get(record.fact_id)
        if existing is not None and not existing.deleted:
            raise FactConflictError(record.fact_id)
        self._items[record.fact_id] = record
        return record

    async def read_fact(self, fact_id: str) -> FactRecord | None:
        return self._items.get(fact_id)

    async def search_facts(self, query: SearchQuery) -> list[FactRecord]:
        if not 1 <= query.limit <= 50:
            raise ValueError("limit")
        filtered = [
            item
            for item in self._items.values()
            if not item.deleted
            and query.text.lower() in item.body.lower()
            and (query.subject is None or item.subject == query.subject)
        ]
        ordered = sorted(filtered, key=lambda item: (-item.created_at.timestamp(), item.fact_id))
        return ordered[: query.limit]

    async def export_markdown(self) -> list[MarkdownDocument]:
        by_subject: dict[str, list[FactRecord]] = {}
        for item in self._items.values():
            if item.deleted:
                continue
            by_subject.setdefault(item.subject, []).append(item)
        documents: list[MarkdownDocument] = []
        for subject, facts in sorted(by_subject.items()):
            ordered = sorted(facts, key=lambda item: (item.created_at, item.fact_id))
            lines = [f"# {subject}"] + [
                f"- {item.created_at.isoformat()} | {item.fact_type} | {item.body}" for item in ordered
            ]
            documents.append(
                MarkdownDocument(
                    path=f"subjects/{subject}.md",
                    content="\n".join(lines) + "\n",
                )
            )
        return documents
