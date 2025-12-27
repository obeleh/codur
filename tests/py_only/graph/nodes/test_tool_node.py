from pathlib import Path

from langchain_core.messages import HumanMessage

from codur.graph.tools import tool_node


class _Runtime:
    allow_outside_workspace = False


class _Config:
    runtime = _Runtime()
    verbose = False


def test_tool_node_adds_python_ast_dependencies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main_py = Path("main.py")
    main_py.write_text("def foo():\n    return bar()\n\ndef bar():\n    return 1\n")

    state = {
        "config": _Config(),
        "messages": [HumanMessage(content="read file")],
        "tool_calls": [{"tool": "read_file", "args": {"path": "main.py"}}],
    }

    result = tool_node(state, _Config())
    summary = result["agent_outcome"]["result"]
    assert "read_file:" in summary
    assert "python_ast_dependencies:" in summary


def test_tool_node_adds_multifile_ast_dependencies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "util.py").write_text("def bar():\n    return 2\n")
    (tmp_path / "notes.txt").write_text("hello\n")

    state = {
        "config": _Config(),
        "messages": [HumanMessage(content="list files")],
        "tool_calls": [{"tool": "list_files", "args": {}}],
    }

    result = tool_node(state, _Config())
    summary = result["agent_outcome"]["result"]
    assert "list_files:" in summary
    assert "python_ast_dependencies_multifile:" in summary
