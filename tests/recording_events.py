"""A test event sink that keeps every count it is given, in order."""
from __future__ import annotations


class RecordingEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def count(self, event: str, **fields) -> None:
        self.events.append((event, fields))
