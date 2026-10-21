"""A Cosmos container fake for CosmosFactBackend's contract tests.

Documents are kept per hierarchical partition key. Transactional batches check every operation first and apply all or
none; a create on an existing id fails with 409 at that operation's index, as the SDK reports it
(azure/cosmos/exceptions.py CosmosBatchOperationError). Replace honours an etag condition with 412. Queries are answered
only for the clauses the adapter defines: an unknown WHERE clause raises AssertionError, so a filter the adapter drops or
misspells fails a test instead of passing silently."""
from __future__ import annotations

import copy
import itertools
import math
import re
from collections.abc import Callable

from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosBatchOperationError,
    CosmosResourceNotFoundError,
)

from graph_rag.adapters import cosmos as adapter


def _pk_of(doc: dict) -> tuple:
    return (doc["family_id"], doc["user_id"], doc["category"])


def _similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if mag == 0.0 else dot / mag


_CLAUSES: dict[str, Callable[[dict, dict], bool]] = {
    adapter.CLAUSE_MEMORY: lambda d, p: "doc_type" not in d,
    adapter.CLAUSE_NOT_DELETED: lambda d, p: d.get("deleted") is not True,
    adapter.CLAUSE_NOT_FORGOTTEN: lambda d, p: d.get("forgotten_at") is None,
    adapter.CLAUSE_NOT_PURGED: lambda d, p: d.get("purged_at") is None,
    adapter.CLAUSE_NOT_EXPIRED: lambda d, p: d.get("valid_until") is None or d["valid_until"] > p["@now"],
    adapter.CLAUSE_STARTED: lambda d, p: d.get("valid_from") is None or d["valid_from"] <= p["@now"],
    adapter.CLAUSE_NOT_EXCLUDED: lambda d, p: d["id"] not in p["@exclude_ids"],
    adapter.CLAUSE_CHANGES: lambda d, p: d.get("doc_type") == "fact_change",
    adapter.CLAUSE_CHANGE_OF_FACT: lambda d, p: d.get("fact_id") == p["@fact_id"],
    adapter.CLAUSE_AFTER_SEQ: lambda d, p: d.get("seq", 0) > p["@after"],
}


class FakeContainer:
    def __init__(self) -> None:
        self.docs: dict[tuple, dict] = {}
        self._etags = itertools.count(1)
        self.before_batch: Callable[[], None] | None = None     # run once, just before the next batch is checked
        self.before_replace: Callable[[], None] | None = None   # run once, just before the next replace is checked
        self.batches: list[list] = []

    def _put(self, doc: dict) -> None:
        stored = copy.deepcopy(doc)
        stored["_etag"] = f"e{next(self._etags)}"
        self.docs[(*_pk_of(doc), doc["id"])] = stored

    def read_item(self, item: str, partition_key: list) -> dict:
        doc = self.docs.get((*partition_key, item))
        if doc is None:
            raise CosmosResourceNotFoundError(status_code=404, message="not found")
        return copy.deepcopy(doc)

    def upsert_item(self, body: dict) -> dict:
        self._put(body)
        return copy.deepcopy(body)

    def replace_item(self, item: str, body: dict, etag: str | None = None, match_condition=None) -> dict:
        if self.before_replace is not None:
            hook, self.before_replace = self.before_replace, None
            hook()
        current = self.docs.get((*_pk_of(body), item))
        if current is None:
            raise CosmosResourceNotFoundError(status_code=404, message="not found")
        if etag is not None and current["_etag"] != etag:
            raise CosmosAccessConditionFailedError(status_code=412, message="precondition failed")
        self._put(body)
        return copy.deepcopy(body)

    def execute_item_batch(self, batch_operations: list, partition_key: list) -> list:
        if self.before_batch is not None:
            hook, self.before_batch = self.before_batch, None
            hook()
        self.batches.append(batch_operations)
        for index, operation in enumerate(batch_operations):
            kind, (doc,) = operation[0], operation[1]
            if kind not in ("create", "upsert"):
                raise AssertionError(f"unsupported batch operation: {kind}")
            if list(_pk_of(doc)) != list(partition_key):
                raise _batch_error(index, 400)
            if kind == "create" and (*partition_key, doc["id"]) in self.docs:
                raise _batch_error(index, 409)
        for operation in batch_operations:
            self._put(operation[1][0])
        return [{"statusCode": 201 if op[0] == "create" else 200} for op in batch_operations]

    def query_items(self, query: str, parameters=None, partition_key=None) -> list[dict]:
        params = {p["name"]: p["value"] for p in (parameters or [])}
        rows = [copy.deepcopy(d) for key, d in self.docs.items() if list(key[:3]) == list(partition_key)]
        select, rest = query.split(" FROM c", 1)
        where = rest.split(" WHERE ", 1)[1] if " WHERE " in rest else ""
        where = where.split(" ORDER BY ", 1)[0]
        for clause in (where.split(" AND ") if where else []):
            if clause not in _CLAUSES:
                raise AssertionError(f"FakeContainer does not know the clause {clause!r}")
            rows = [d for d in rows if _CLAUSES[clause](d, params)]
        if "ORDER BY c.seq ASC" in query:
            rows.sort(key=lambda d: d["seq"])
        if "VectorDistance" in select:
            rows = [d for d in rows if d.get("vector_embeddings")]
            for d in rows:
                d["score"] = _similarity(params["@qv"], d["vector_embeddings"])
            rows.sort(key=lambda d: -d["score"])
            fields = [part[2:] for part in select.split(" ", 3)[3].split(", ") if part.startswith("c.")]
            rows = [{**{f: d[f] for f in fields if f in d}, "score": d["score"]} for d in rows]
        top = re.match(r"SELECT TOP (@\w+|\d+) ", query)
        if top:
            rows = rows[: params[top[1]] if top[1].startswith("@") else int(top[1])]
        return rows


def _batch_error(index: int, status: int) -> CosmosBatchOperationError:
    return CosmosBatchOperationError(error_index=index, headers={}, status_code=status,
                                     message=f"batch operation {index} failed", operation_responses=[])
