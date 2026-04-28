from datetime import datetime

import pytest

from graph_rag.adapters.in_memory import InMemoryFactStore
from graph_rag.contracts import FactRecord, SearchQuery
from graph_rag.exceptions import FactConflictError


def _record(fact_id: str, body: str, subject: str, created_at: str, deleted: bool = False) -> FactRecord:
    dt = datetime.fromisoformat(created_at)
    return FactRecord(
        fact_id=fact_id,
        body=body,
        subject=subject,
        fact_type="preference",
        confidence=0.9,
        created_at=dt,
        updated_at=dt,
        valid_until=None,
        source="manual",
        deleted=deleted,
    )


@pytest.mark.asyncio
async def test_in_memory_create_read_search_and_export():
    store = InMemoryFactStore()
    await store.create_fact(_record("fact-001", "child1 likes blueberries", "child1", "2026-04-01T09:00:00+00:00"))
    await store.create_fact(_record("fact-002", "child1 likes strawberries", "child1", "2026-04-02T09:00:00+00:00"))
    assert await store.read_fact("fact-001") is not None
    results = await store.search_facts(SearchQuery(text="likes", subject="child1", limit=10))
    assert [item.fact_id for item in results] == ["fact-002", "fact-001"]
    exported = await store.export_markdown()
    assert exported[0].path == "subjects/child1.md"
    assert exported[0].content == (
        "# child1\n"
        "- 2026-04-01T09:00:00+00:00 | preference | child1 likes blueberries\n"
        "- 2026-04-02T09:00:00+00:00 | preference | child1 likes strawberries\n"
    )


@pytest.mark.asyncio
async def test_in_memory_read_missing_returns_none():
    store = InMemoryFactStore()
    assert await store.read_fact("missing-id") is None


@pytest.mark.asyncio
async def test_in_memory_duplicate_create_raises_conflict():
    store = InMemoryFactStore()
    record = _record("fact-001", "child1 likes blueberries", "child1", "2026-04-01T09:00:00+00:00")
    await store.create_fact(record)
    with pytest.raises(FactConflictError):
        await store.create_fact(record)
    current = await store.read_fact("fact-001")
    assert current == record
    assert current is not None
    assert current.deleted is False


@pytest.mark.asyncio
async def test_in_memory_rejects_invalid_limit_and_excludes_deleted():
    store = InMemoryFactStore()
    await store.create_fact(_record("fact-004", "child1 likes pears", "child1", "2026-04-04T09:00:00+00:00", deleted=True))
    with pytest.raises(ValueError, match="limit"):
        await store.search_facts(SearchQuery(text="likes", subject=None, limit=0))
    results = await store.search_facts(SearchQuery(text="likes", subject=None, limit=10))
    assert results == []
