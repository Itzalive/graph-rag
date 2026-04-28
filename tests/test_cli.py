from graph_rag.cli import main


def test_cli_check_uses_local_defaults(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["check"]) == 0
    assert "backend=sqlite" in capsys.readouterr().out


def test_cli_rejects_invalid_backend(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_RAG_BACKEND", "postgres")
    assert main(["check"]) == 1
    assert "Unsupported backend" in capsys.readouterr().err


def test_cli_export_markdown_prints_subject_paths(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_RAG_BACKEND", "memory")
    assert main(["export-markdown"]) == 0
    assert capsys.readouterr().out == ""
