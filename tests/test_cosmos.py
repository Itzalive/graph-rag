from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, call, patch

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.facts import FactRecord, FactSearchRequest, Holder


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


def test_constructing_from_a_url_builds_no_client():
    # claude-assistant-5si8: the CosmosClient constructor fetches the account over the network; not on the event loop.
    with patch("azure.cosmos.CosmosClient") as client_cls:
        CosmosFactBackend(cosmos_url="https://x.documents.azure.com", credential=object(), database_name="db", family_id="f")
    client_cls.assert_not_called()


async def test_the_first_operation_builds_one_client_in_a_worker_thread():
    import asyncio
    import threading
    import time

    on_main = []

    def build(**kwargs):
        on_main.append(threading.current_thread() is threading.main_thread())
        time.sleep(0.05)   # the account fetch: the other reads arrive while it runs
        client = MagicMock()
        client.get_database_client.return_value.get_container_client.return_value.read_item.return_value = {}
        return client

    with patch("azure.cosmos.CosmosClient", side_effect=build) as client_cls:
        backend = CosmosFactBackend(cosmos_url="https://x.documents.azure.com", credential=object(), database_name="db",
                                    family_id="f")
        with patch.object(CosmosFactBackend, "_from_document", return_value=None):
            await asyncio.gather(*(backend.read_fact(str(i)) for i in range(3)))
    assert on_main == [False]
    assert client_cls.call_count == 1


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
    assert "ORDER BY VectorDistance(c.vector_embeddings, @qv, false)" in query


async def test_search_sql_contains_exclude_ids_filter():
    backend, container = _backend()
    container.query_items.return_value = iter([])
    await backend.search_facts(
        FactSearchRequest(query_vector=(1.0, 0.0), limit=5, exclude_ids=("f1", "f2"))
    )
    query: str = container.query_items.call_args[1]["query"]
    params = {p["name"]: p["value"] for p in container.query_items.call_args[1]["parameters"]}
    assert "ARRAY_CONTAINS(@exclude_ids, c.id)" in query
    assert params["@exclude_ids"] == ["f1", "f2"]


async def test_search_score_is_cosine_distance_from_the_similarity_cosmos_returns():
    """Cosmos VectorDistance with the cosine function returns a similarity (1 = identical), most similar first."""
    backend, container = _backend()
    container.query_items.return_value = iter([
        {"id": "f1", "fact": "close", "score": 0.9, "created_at": datetime.now(UTC).isoformat(), "extra": "kept"},
        {"id": "f2", "fact": "far", "score": 0.05, "created_at": datetime.now(UTC).isoformat()},
    ])
    hits = await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=5))
    query: str = container.query_items.call_args[1]["query"]
    assert "AS score" in query
    assert [hit.fact_id for hit in hits] == ["f1", "f2"]
    assert hits[0].score == pytest.approx(0.1)
    assert hits[1].score == pytest.approx(0.95)
    assert hits[0].extras == {"extra": "kept"}


async def test_search_missing_score_defaults_to_infinite_distance():
    backend, container = _backend()
    container.query_items.return_value = iter([
        {"id": "f1", "fact": "body", "created_at": datetime.now(UTC).isoformat()},
    ])
    hits = await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=5))
    assert hits[0].score == float("inf")


async def test_search_never_selects_the_embedding_vector():
    """Each vector is ~22 KB of JSON: a top-20 search must not download 20 of them only to drop them (claude-assistant-yh9p)."""
    backend, container = _backend()
    container.query_items.return_value = iter([])
    await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=20))
    query: str = container.query_items.call_args[1]["query"]
    select_list = query.split(" FROM ", 1)[0].replace("VectorDistance(c.vector_embeddings, @qv, false) AS score", "")
    assert not select_list.startswith("SELECT TOP 20 c,")   # the whole document
    assert "vector_embeddings" not in select_list          # only the score expression may read the vector
    for field in ("c.id", "c.fact", "c.created_at", "c.last_updated_at", "c.source", "c.subject", "c.fact_type",
                  "c.confidence", "c.valid_until", "c.extracted_at", "c.history", "c.metadata", "c.updated_at", "c.user",
                  "c.kind", "c.holder", "c.lifespan", "c.valid_from", "c.validity_basis", "c.forgotten_at", "c.purged_at",
                  "c.learned_at", "c.last_changed_at", "c.last_confirmed_at", "c.entities", "c.topics", "c.seq",
                  "c.constraint", "c.situation", "c.attribute", "c.calc"):
        assert field in select_list, field


async def test_upsert_writes_recorded_fields_top_level():
    backend, container = _backend()
    now = datetime.now(UTC)
    await backend.upsert_fact(FactRecord(fact_id="m1", body="b", created_at=now, updated_at=now, kind="opinion",
                                         holder=Holder("parent", "parent2"), topics=("people",), seq=2,
                                         extras={"situation": {"channels": ["chat"]}}))
    doc = container.upsert_item.call_args[0][0]
    assert (doc["kind"], doc["holder"], doc["topics"], doc["seq"], doc["situation"]) == (
        "opinion", {"kind": "parent", "ref": "parent2"}, ["people"], 2, {"channels": ["chat"]})


async def test_read_returns_recorded_fields_typed_and_not_in_extras():
    backend, container = _backend()
    now = datetime.now(UTC).isoformat()
    container.read_item.return_value = {"id": "m1", "fact": "b", "created_at": now, "kind": "fact", "seq": 4,
                                        "holder": {"kind": "kerwen"}, "entities": ["pet-1"], "calc": {"rule": "age"}}
    record = await backend.read_fact("m1")
    assert (record.kind, record.seq, record.holder, record.entities) == ("fact", 4, Holder("kerwen"), ("pet-1",))
    assert record.extras == {"calc": {"rule": "age"}}
