from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

from graph_rag.contracts import FactRecord, MarkdownDocument, SearchQuery
from graph_rag.exceptions import FactConflictError


class SQLiteFactStore:
    def __init__(self, database_path: Path | str) -> None:
        self._database_path = Path(database_path)
        self._schema_ready = False

    async def create_fact(self, record: FactRecord) -> FactRecord:
        await self._ensure_schema()
        return await asyncio.to_thread(self._create_fact_sync, record)

    async def read_fact(self, fact_id: str) -> FactRecord | None:
        await self._ensure_schema()
        return await asyncio.to_thread(self._read_fact_sync, fact_id)

    async def search_facts(self, query: SearchQuery) -> list[FactRecord]:
        if not 1 <= query.limit <= 50:
            raise ValueError("limit")
        await self._ensure_schema()
        return await asyncio.to_thread(self._search_facts_sync, query)

    async def export_markdown(self) -> list[MarkdownDocument]:
        await self._ensure_schema()
        return await asyncio.to_thread(self._export_markdown_sync)

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        await asyncio.to_thread(self._ensure_schema_sync)
        self._schema_ready = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema_sync(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    body TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    valid_until TEXT,
                    source TEXT NOT NULL,
                    deleted INTEGER NOT NULL
                )
                """
            )

    def _create_fact_sync(self, record: FactRecord) -> FactRecord:
        if not 0.0 <= record.confidence <= 1.0:
            raise ValueError("confidence")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT deleted FROM facts WHERE fact_id = ?",
                (record.fact_id,),
            ).fetchone()
            if existing is not None and not bool(existing["deleted"]):
                raise FactConflictError(record.fact_id)
            connection.execute(
                """
                INSERT OR REPLACE INTO facts
                (fact_id, body, subject, fact_type, confidence, created_at, updated_at, valid_until, source, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.fact_id,
                    record.body,
                    record.subject,
                    record.fact_type,
                    record.confidence,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.valid_until.isoformat() if record.valid_until else None,
                    record.source,
                    int(record.deleted),
                ),
            )
        return record

    def _read_fact_sync(self, fact_id: str) -> FactRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        return None if row is None else self._row_to_record(row)

    def _search_facts_sync(self, query: SearchQuery) -> list[FactRecord]:
        like_value = f"%{self._escape_like(query.text.lower())}%"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM facts
                WHERE deleted = 0
                  AND lower(body) LIKE ? ESCAPE '\\'
                  AND (? IS NULL OR subject = ?)
                ORDER BY created_at DESC, fact_id ASC
                LIMIT ?
                """,
                (like_value, query.subject, query.subject, query.limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _escape_like(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _export_markdown_sync(self) -> list[MarkdownDocument]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM facts
                WHERE deleted = 0
                ORDER BY subject ASC, created_at ASC, fact_id ASC
                """
            ).fetchall()
        by_subject: dict[str, list[FactRecord]] = {}
        for row in rows:
            record = self._row_to_record(row)
            by_subject.setdefault(record.subject, []).append(record)
        return [
            MarkdownDocument(
                path=f"subjects/{subject}.md",
                content="\n".join(
                    [f"# {subject}"]
                    + [f"- {item.created_at.isoformat()} | {item.fact_type} | {item.body}" for item in facts]
                )
                + "\n",
            )
            for subject, facts in by_subject.items()
        ]

    def _row_to_record(self, row: sqlite3.Row) -> FactRecord:
        return FactRecord(
            fact_id=row["fact_id"],
            body=row["body"],
            subject=row["subject"],
            fact_type=row["fact_type"],
            confidence=float(row["confidence"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            source=row["source"],
            deleted=bool(row["deleted"]),
        )
