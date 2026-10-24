from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from graph_rag.entities import EntityRecord
from graph_rag.facts import FactRecord
from graph_rag.record import (
    ChangeSource, FactChange, WriteCheckError, change_from_state, change_id, change_to_state, check_write, op_marker_id,
)

T = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)
SRC = ChangeSource("parent_chat", actor="parent1", ref="chat:parent1:2026-10-18T09:00:00+00:00")


def _rec(**fields) -> FactRecord:
    base = dict(fact_id="m1", body="b", created_at=T, updated_at=T, lifespan="lasting", seq=1)
    return FactRecord(**{**base, **fields})


def _change(seq: int = 1, **fields) -> FactChange:
    base = dict(fact_id="m1", seq=seq, kind="created", at=T, source=SRC, reason="told in chat", op_id="tool-1")
    return FactChange(**{**base, **fields})


def test_a_valid_write_passes():
    check_write(_rec(), [_change()])


@pytest.mark.parametrize("record, code", [
    (_rec(lifespan="dated"), "W1"),
    (_rec(valid_until=T + timedelta(days=1), validity_basis="stated"), "W2"),
    (_rec(lifespan="dated", valid_from=T, valid_until=T + timedelta(days=1)), "W3"),
    (_rec(seq=2), "W4"),
])
def test_each_write_check_refuses_with_its_code(record, code):
    with pytest.raises(WriteCheckError) as exc:
        check_write(record, [_change()])
    assert exc.value.code == code


def test_ended_basis_allows_valid_until_on_any_lifespan():
    check_write(_rec(lifespan="fading", valid_until=T, validity_basis="ended"), [_change()])


@pytest.mark.parametrize("changes, message", [
    ([], "one or two"),
    ([_change(0, after={}), _change(1), _change(2)], "one or two"),
    ([_change(fact_id="other")], "fact_id"),
    ([_change(0, after={"id": "m1"}), _change(2)], "consecutive"),
    ([_change(1, after={"id": "m1"}), _change(2)], "seq 0"),
    ([_change(op_id="")], "op_id"),
    ([_change(after={"id": "m1"})], "after"),
    ([_change(0), _change(1)], "after"),
    ([_change(op_id="email:a/b")], "must not contain"),
])
def test_malformed_change_lists_are_refused(changes, message):
    record = _rec(seq=changes[-1].seq if changes else 1)
    with pytest.raises(ValueError, match=message):
        check_write(record, changes)


def test_change_document_round_trips():
    change = _change(after={"id": "m1", "fact": "b"}, approved_by="parent1",
                     source=ChangeSource("rebuild", actor="parent1", ref="run-7", label="chat"))
    state = change_to_state(change)
    assert (state["id"], state["doc_type"]) == (change_id("m1", 1), "fact_change")
    assert change_from_state(state) == change


def test_ids():
    assert change_id("m1", 4) == "m1:4"
    assert op_marker_id("email:abc:9f") == "op-email:abc:9f"


PARK = EntityRecord(entity_id="place-1", type="place", name="Park", handle="park", version="e3")


def _parks(n: int, *, start: int = 0) -> list[EntityRecord]:
    return [replace(PARK, entity_id=f"place-{i}") for i in range(start, start + n)]


def test_entity_updates_count_toward_the_100_operations():
    check_write(_rec(), [_change()], entity_updates=_parks(97))           # marker + change + memory + 97 = 100
    with pytest.raises(ValueError, match="100"):
        check_write(_rec(), [_change()], entities=_parks(1, start=200), entity_updates=_parks(97))


@pytest.mark.parametrize("entities, updates, message", [
    ([], [replace(PARK, version=None)], "version"),
    ([replace(PARK, version=None)], [PARK], "distinct"),
    ([], [PARK, PARK], "distinct"),
])
def test_malformed_entity_updates_are_refused(entities, updates, message):
    with pytest.raises(ValueError, match=message):
        check_write(_rec(), [_change()], entities=entities, entity_updates=updates)
