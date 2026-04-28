from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass
class Settings:
    env: str = "local"
    backend: str = "sqlite"
    database_path: str = "data/graph_rag.sqlite3"
    log_level: str = "INFO"


def load_settings() -> Settings:
    env_local = Path(".env.local")
    env_local_values = dotenv_values(env_local) if env_local.exists() else {}

    backend = os.getenv("GRAPH_RAG_BACKEND", env_local_values.get("GRAPH_RAG_BACKEND", "sqlite"))
    if backend not in {"memory", "sqlite"}:
        raise ValueError(f"Unsupported backend: {backend}")

    return Settings(
        env=os.getenv("GRAPH_RAG_ENV", env_local_values.get("GRAPH_RAG_ENV", "local")),
        backend=backend,
        database_path=os.getenv(
            "GRAPH_RAG_DATABASE_PATH",
            env_local_values.get("GRAPH_RAG_DATABASE_PATH", "data/graph_rag.sqlite3"),
        ),
        log_level=os.getenv("LOG_LEVEL", env_local_values.get("LOG_LEVEL", "INFO")),
    )
