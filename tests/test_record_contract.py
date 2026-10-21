"""The recorded-memory contract (claude-assistant-xz59.9 §2–§4), run against both adapters."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.facts import FactRecord, FactSearchRequest
from cosmos_fake import FakeContainer  # tests/ is on sys.path (no __init__.py; pytest prepend import mode)

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _rec(fact_id: str = "m1", body: str = "Alice swims on Tuesdays", **fields) -> FactRecord:
    return FactRecord(fact_id=fact_id, body=body, created_at=NOW, updated_at=NOW, **fields)


@pytest.fixture(params=["memory", "cosmos"])
def make_backend(request):
    """A factory for backends sharing one store: for Cosmos one fake container, for memory one bucket per instance."""
    container = FakeContainer()

    def make(family_id: str = "fam1", scope: str = "shared"):
        if request.param == "memory":
            return InMemoryFactBackend(clock=lambda: NOW)
        return CosmosFactBackend(container=container, family_id=family_id, scope=scope, clock=lambda: NOW)

    make.kind = request.param
    make.container = container
    return make


async def test_two_families_never_see_each_others_memories(make_backend):   # AC-M6
    fam1, fam2 = make_backend("fam1"), make_backend("fam2")
    await fam1.upsert_fact(_rec(vector_embeddings=(1.0, 0.0)))
    assert await fam2.read_fact("m1") is None
    assert await fam2.list_facts() == []
    assert await fam2.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=5)) == []
    assert [r.fact_id for r in await fam1.list_facts()] == ["m1"]


async def test_cosmos_scope_names_the_bucket():
    container = FakeContainer()
    personal = CosmosFactBackend(container=container, family_id="fam1", scope="parent1", clock=lambda: NOW)
    await personal.upsert_fact(_rec())
    assert list(container.docs) == [("fam1", "parent1", "FACTS", "m1")]
    assert container.docs[("fam1", "parent1", "FACTS", "m1")]["is_shared"] is False
    shared = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    assert await shared.read_fact("m1") is None


def test_cosmos_scope_must_not_be_empty():
    with pytest.raises(ValueError, match="scope"):
        CosmosFactBackend(container=FakeContainer(), family_id="fam1", scope="")


async def _ids(backend, **flags) -> tuple[list[str], list[str]]:
    listed = sorted(r.fact_id for r in await backend.list_facts(**flags))
    request = FactSearchRequest(query_vector=(1.0, 0.0), limit=10, **flags)
    searched = sorted(h.fact_id for h in await backend.search_facts(request))
    return listed, searched


async def test_forgotten_and_purged_are_never_listed_or_searched(make_backend):   # AC-M5
    backend = make_backend()
    vec = (1.0, 0.0)
    await backend.upsert_fact(_rec("live", vector_embeddings=vec))
    await backend.upsert_fact(_rec("forgot", forgotten_at=NOW, vector_embeddings=vec))
    await backend.upsert_fact(_rec("purged", purged_at=NOW, vector_embeddings=vec))
    assert await _ids(backend) == (["live"], ["live"])
    assert await _ids(backend, include_expired=True, include_future=True) == (["live"], ["live"])


async def test_not_yet_valid_only_with_include_future(make_backend):   # AC-M5, AC-14
    backend = make_backend()
    vec = (1.0, 0.0)
    await backend.upsert_fact(_rec("now", valid_from=NOW - timedelta(days=1), validity_basis="stated",
                                   vector_embeddings=vec))
    await backend.upsert_fact(_rec("later", valid_from=NOW + timedelta(days=16), validity_basis="stated",
                                   vector_embeddings=vec))
    assert await _ids(backend) == (["now"], ["now"])
    assert await _ids(backend, include_future=True) == (["later", "now"], ["later", "now"])


async def test_past_valid_until_only_with_include_expired(make_backend):   # AC-M5
    backend = make_backend()
    vec = (1.0, 0.0)
    await backend.upsert_fact(_rec("ended", valid_until=NOW - timedelta(days=1), validity_basis="ended",
                                   vector_embeddings=vec))
    assert await _ids(backend) == ([], [])
    assert await _ids(backend, include_expired=True) == (["ended"], ["ended"])


async def test_cosmos_documents_with_a_doc_type_are_never_memories():   # AC-M5 (read_fact on a non-memory id)
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    container.upsert_item({"id": "x:1", "doc_type": "fact_change", "fact": "copy", "family_id": "fam1",
                           "user_id": "shared", "category": "FACTS", "vector_embeddings": [1.0, 0.0]})
    assert await backend.read_fact("x:1") is None
    assert await _ids(backend, include_expired=True, include_future=True) == ([], [])
