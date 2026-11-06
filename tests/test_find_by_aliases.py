"""find_by_aliases (claude-assistant-xz59.13.8 §2.2), run against both recorded backends."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.entities import EntityRecord
from cosmos_fake import FakeContainer
from recording_events import RecordingEvents

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


@dataclass
class Bucket:
    backend: object
    events: RecordingEvents
    container: FakeContainer | None


@pytest.fixture(params=["memory", "cosmos"])
def bucket(request) -> Bucket:
    events = RecordingEvents()
    if request.param == "memory":
        return Bucket(InMemoryFactBackend(clock=lambda: NOW, events=events), events, None)
    container = FakeContainer()
    return Bucket(CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW, events=events),
                  events, container)


def _entity(entity_id: str, name: str, *aliases: str, type: str = "person") -> EntityRecord:
    return EntityRecord(entity_id=entity_id, type=type, name=name, handle=entity_id.replace("-", "_"), aliases=aliases)


async def _seed(bucket: Bucket, *entities: EntityRecord) -> None:
    for entity in entities:
        await bucket.backend.upsert_entity(entity)


async def _ids(bucket: Bucket, keys, **kwargs) -> list[str]:
    return [e.entity_id for e in await bucket.backend.find_by_aliases(keys, **kwargs)]


async def test_a_key_finds_its_entity_among_more_than_any_read_limit(bucket):   # AC-R25b
    await _seed(bucket, *[_entity(f"person-{i:03}", f"Person {i:03}") for i in range(350)])
    assert await _ids(bucket, ["PERSON 217"]) == ["person-217"]
    assert bucket.events.events == []


async def test_any_type_matches_and_names_and_aliases_both_count(bucket):
    await _seed(bucket, _entity("org-1", "Oakfield School", "Oakfield", type="organisation"),
                _entity("person-1", "Ann", "Annie"))
    assert await _ids(bucket, ["oakfield", "annie"]) == ["org-1", "person-1"]


async def test_results_rank_longest_matching_key_then_id_and_cut_to_limit(bucket):   # AC-S1-4
    await _seed(bucket,
                _entity("person-c", "Sam", "Samantha Jones"),     # longest key it has: 14, but only "sam" matches
                _entity("person-b", "Sam Jones"),                 # matches "sam jones": 9
                _entity("person-a", "Sam"),                       # matches "sam": 3
                _entity("person-d", "Sammy", "Sam"))              # matches "sam": 3
    assert await _ids(bucket, ["sam", "sam jones"]) == ["person-b", "person-a", "person-c", "person-d"]
    assert await _ids(bucket, ["sam", "sam jones"], limit=2) == ["person-b", "person-a"]
    assert bucket.events.events == [("aliases.limit_reached", {"limit": 2})]


@pytest.mark.parametrize("keys, given", [
    ([f"k{i}" for i in range(301)], 301),
    ([f"k{i}" for i in range(301)] + ["K0", "k1", ""], 301),
], ids=["301", "duplicates"])
async def test_past_300_distinct_keys_only_the_first_300_are_used(bucket, keys, given):   # AC-R25b
    await _seed(bucket, _entity("person-1", "k299"), _entity("person-2", "k300"))
    assert await _ids(bucket, keys) == ["person-1"]
    assert bucket.events.events == [("aliases.keys_capped", {"given": given, "used": 300})]


async def test_300_distinct_keys_raise_no_cap(bucket):
    await _seed(bucket, _entity("person-1", "k299"))
    assert await _ids(bucket, [f"k{i}" for i in range(300)]) == ["person-1"]
    assert bucket.events.events == []


async def test_more_than_300_matches_keep_300_and_count_it(bucket):
    await _seed(bucket, *[_entity(f"person-{i:03}", "Sam") for i in range(301)])
    found = await _ids(bucket, ["sam"], limit=50)
    assert len(found) == 50 and len(set(found)) == 50
    assert [e for e, _ in bucket.events.events] == ["aliases.read_capped", "aliases.limit_reached"]
    if bucket.container is None:                      # in memory the 300 lowest ids are read
        assert found == [f"person-{i:03}" for i in range(50)]


async def test_no_keys_reads_nothing(bucket):   # AC-S1-4
    if bucket.container is not None:
        bucket.container.query_items = _no_read
    assert await bucket.backend.find_by_aliases([]) == []
    assert await bucket.backend.find_by_aliases(["", ""]) == []


@pytest.mark.parametrize("limit", [0, 51])
async def test_limit_outside_1_to_50_is_refused_before_any_read(bucket, limit):   # AC-S1-5
    if bucket.container is not None:
        bucket.container.query_items = _no_read
    with pytest.raises(ValueError):
        await bucket.backend.find_by_aliases(["sam"], limit=limit)


@pytest.mark.parametrize("limit", [1, 50])
async def test_limits_1_and_50_are_accepted(bucket, limit):
    await _seed(bucket, *[_entity(f"person-{i:02}", "Sam") for i in range(limit)])   # exactly `limit` matches
    assert len(await _ids(bucket, ["sam"], limit=limit)) == limit
    assert bucket.events.events == []


async def test_exactly_300_matches_raise_no_read_cap(bucket):
    await _seed(bucket, *[_entity(f"person-{i:03}", "Sam") for i in range(300)])
    assert len(await _ids(bucket, ["sam"], limit=50)) == 50
    assert bucket.events.events == [("aliases.limit_reached", {"limit": 50})]


def _no_read(*args, **kwargs):
    raise AssertionError("a refused call must not read")


class _Spy(FakeContainer):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[tuple[str, dict]] = []

    def query_items(self, query, parameters=None, partition_key=None):
        self.queries.append((query, {p["name"]: p["value"] for p in parameters or []}))
        return super().query_items(query, parameters=parameters, partition_key=partition_key)


async def test_cosmos_alias_keys_are_one_parameter_never_in_the_query():   # trust boundary row 2
    container = _Spy()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.find_by_aliases(["O'Brien", 'Ann "Nan"', "o'brien"])
    [(query, params)] = container.queries
    assert query == ("SELECT TOP 301 * FROM c WHERE c.doc_type = 'entity' AND "
                     "EXISTS(SELECT VALUE k FROM k IN c.alias_keys WHERE ARRAY_CONTAINS(@keys, k))")
    assert params == {"@keys": ["o'brien", 'ann "nan"']}


async def test_every_limit_event_carries_only_its_name_integers_and_fixed_words(bucket):   # AC-S1-9
    from datetime import timedelta  # noqa: PLC0415
    from graph_rag.facts import FactRecord, Situation  # noqa: PLC0415

    secret_name = "Zebedee Quux"
    await _seed(bucket, *[_entity(f"person-{i:03}", secret_name) for i in range(301)])
    await bucket.backend.find_by_aliases([secret_name, *[f"other {i}" for i in range(300)]], limit=1)
    for i in range(3):
        await bucket.backend.upsert_fact(FactRecord(
            fact_id=f"m{i}", body="Zebedee's secret wording", created_at=NOW, updated_at=NOW, kind="instruction",
            situation=Situation(("email",)), learned_at=NOW - timedelta(days=i)))
    await bucket.backend.situational("email", [], [], unsituated_kinds=("instruction",), limit=1)
    names = [event for event, _ in bucket.events.events]
    assert names == ["aliases.keys_capped", "aliases.read_capped", "aliases.limit_reached",
                     "situational.limit_reached"]
    for _, fields in bucket.events.events:
        for value in fields.values():
            assert isinstance(value, int) or value in {"situated", "unsituated"}, value
