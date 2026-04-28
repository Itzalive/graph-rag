from pathlib import Path

import pytest

from graph_rag.config import load_settings


def test_load_settings_defaults_to_local_sqlite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.env == "local"
    assert settings.backend == "sqlite"
    assert settings.database_path == "data/graph_rag.sqlite3"


def test_load_settings_reads_env_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path(".env.local").write_text("GRAPH_RAG_BACKEND=memory\nLOG_LEVEL=DEBUG\n")
    settings = load_settings()
    assert settings.backend == "memory"
    assert settings.log_level == "DEBUG"


def test_load_settings_rejects_invalid_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path(".env.local").write_text("GRAPH_RAG_BACKEND=postgres\n")
    with pytest.raises(ValueError, match="backend"):
        load_settings()
