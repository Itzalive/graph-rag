"""Counts a backend raises when a read or write reaches one of its limits. graph-rag
passes an event name and fields that are integers or its own fixed words, never memory wording, alias keys, ids or
names. A sink that raises is the caller's bug: graph-rag does not catch it."""
from __future__ import annotations

from typing import Protocol


class EventSink(Protocol):
    def count(self, event: str, **fields: str | int) -> None: ...


class _NoEvents:
    def count(self, event: str, **fields: str | int) -> None:
        return None


NO_EVENTS: EventSink = _NoEvents()
