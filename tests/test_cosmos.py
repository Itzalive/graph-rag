from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, call, patch

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.facts import FactRecord, FactSearchRequest


def _backend(family_id: str = "fam1") -> tuple[CosmosFactBackend, MagicMock]:
    container = MagicMock()
    container.query_items.return_value = iter([])
    return CosmosFactBackend(container=container, family_id=family_id), container


def _record(fact_id: str = "f1", *, deleted: bool = False) -> FactRecord:
    now = datetime.now(UTC)
    return FactRecord(
        fact_id=fact_id,
        body="child likes blueberries",
        created_at=now,
        updated_at=now,
        deleted=deleted,
        source="manual",
        subject="child1",
        fact_type="preference",
        confidence=0.9,
    )


# --- constructor ---


def test_missing_constructor_args_raises():
    with pytest.raises(ValueError):
        CosmosFactBackend(family_id="fam1")


def test_constructor_with_container_does_not_import_cosmos():
    container = MagicMock()
    backend = CosmosFactBackend(container=container, family_id="fam1")
    assert backend is not None


# --- upsert ---


async def test_upsert_fact_writes_correctly_shaped_document():
    backend, container = _backend()
    container.upsert_item.return_value = None
    rec = _record()
    await backend.upsert_fact(rec)
    container.upsert_item.assert_called_once()
    doc = container.upsert_item.call_args[0][0]
    assert doc["id"] == "f1"
    assert doc["fact"] == "child likes blueberries"
    assert doc["family_id"] == "fam1"
    assert doc["user_id"] == "shared"
    assert doc["category"] == "FACTS"
    assert doc["is_shared"] is True


# --- read ---


async def test_read_fact_uses_partitioned_read_item():
    backend, container = _backend("fam1")
    now = datetime.now(UTC).isoformat()
    container.read_item.return_value = {
        "id": "f1",
        "fact": "body",
        "created_at": now,
        "last_updated_at": now,
    }
    result = await backend.read_fact("f1")
    container.read_item.assert_called_once_with(
        item="f1",
        partition_key=["fam1", "shared", "FACTS"],
    )
    assert result.fact_id == "f1"


# --- list ---


async def test_list_facts_builds_correct_filters():
    backend, container = _backend()
    container.query_items.return_value = iter([])
    await backend.list_facts()
    query: str = container.query_items.call_args[1]["query"]
    assert "c.deleted" in query
    assert "SELECT" in query


# --- search ---


async def test_search_rejects_limit_over_100():
    backend, _ = _backend()
    with pytest.raises(ValueError, match="100"):
        await backend.search_facts(FactSearchRequest(query_vector=(1.0,), limit=101))


async def test_search_sql_contains_deleted_filter():
    backend, container = _backend()
    container.query_items.return_value = iter([])
    await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=5))
    query: str = container.query_items.call_args[1]["query"]
    assert "c.deleted" in query
    assert "VectorDistance" in query
