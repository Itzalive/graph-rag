from datetime import datetime

import pytest

from graph_rag.adapters.sqlite import SQLiteFactStore
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
async def test_sqlite_matches_contract_and_export(tmp_path):
    store = SQLiteFactStore(tmp_path / "graph_rag.sqlite3")
    await store.create_fact(_record("fact-001", "child1 likes blueberries", "child1", "2026-04-01T09:00:00+00:00"))
    await store.create_fact(_record("fact-002", "child1 likes strawberries", "child1", "2026-04-02T09:00:00+00:00"))
    await store.create_fact(_record("fact-003", "child1 likes pears", "child1", "2026-04-03T09:00:00+00:00", deleted=True))
    results = await store.search_facts(SearchQuery(text="likes", subject="child1", limit=10))
    assert [item.fact_id for item in results] == ["fact-002", "fact-001"]
    exported = await store.export_markdown()
    assert "pears" not in exported[0].content
    assert exported[0].content.endswith("child1 likes strawberries\n")


@pytest.mark.asyncio
async def test_sqlite_read_missing_returns_none(tmp_path):
    store = SQLiteFactStore(tmp_path / "graph_rag.sqlite3")
    assert await store.read_fact("missing-id") is None


@pytest.mark.asyncio
async def test_sqlite_duplicate_create_raises_conflict(tmp_path):
    store = SQLiteFactStore(tmp_path / "graph_rag.sqlite3")
    record = _record("fact-001", "child1 likes blueberries", "child1", "2026-04-01T09:00:00+00:00")
    await store.create_fact(record)
    with pytest.raises(FactConflictError):
        await store.create_fact(record)
    current = await store.read_fact("fact-001")
    assert current == record
    assert current is not None
    assert current.deleted is False


@pytest.mark.asyncio
async def test_sqlite_uses_asyncio_to_thread(monkeypatch, tmp_path):
    calls: list[str] = []
    original_to_thread = __import__("asyncio").to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        calls.append(func.__name__)
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("graph_rag.adapters.sqlite.asyncio.to_thread", tracking_to_thread)
    store = SQLiteFactStore(tmp_path / "graph_rag.sqlite3")
    await store.create_fact(_record("fact-001", "child1 likes blueberries", "child1", "2026-04-01T09:00:00+00:00"))
    await store.search_facts(SearchQuery(text="likes", subject="child1", limit=10))
    assert calls
