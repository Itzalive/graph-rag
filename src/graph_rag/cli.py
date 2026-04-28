from __future__ import annotations

import argparse
import asyncio
import sys

from graph_rag.adapters.in_memory import InMemoryFactStore
from graph_rag.adapters.sqlite import SQLiteFactStore
from graph_rag.config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graph_rag")
    parser.add_argument("command", choices=["check", "export-markdown"])
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    store = InMemoryFactStore() if settings.backend == "memory" else SQLiteFactStore(settings.database_path)
    if args.command == "check":
        print(f"backend={settings.backend} database_path={settings.database_path}")
        return 0

    exported = asyncio.run(store.export_markdown())
    for document in exported:
        print(document.path)
        print(document.content, end="")
    return 0
