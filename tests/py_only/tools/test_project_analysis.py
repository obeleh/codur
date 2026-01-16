"""Tests for project analysis tools."""

from pathlib import Path

import pytest

from codur.tools.project_analysis import python_dependency_graph, code_quality


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_python_dependency_graph_edges_and_excludes(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _write(root / "pkg" / "__init__.py", "from .mod import foo\n")
    _write(root / "pkg" / "mod.py", "import os\nfrom .sub import bar\n\ndef foo():\n    return bar()\n")
    _write(root / "pkg" / "sub.py", "def bar():\n    return 1\n")
    _write(root / "main.py", "from pkg import foo\n\nfoo()\n")
    _write(root / "ignored" / "extra.py", "import sys\n")
    _write(root / "bad.py", "def broken(:\n    pass\n")

    graph = python_dependency_graph(
        root=root,
        include_external=True,
        exclude_folders=["ignored"],
    )

    nodes = set(graph["nodes"])
    assert "pkg" in nodes
    assert "pkg.mod" in nodes
    assert "pkg.sub" in nodes
    assert "main" in nodes
    assert "ignored.extra" not in nodes
    assert "os" in graph["external_nodes"]

    edges = {(edge["from"], edge["to"]) for edge in graph["edges"]}
    assert ("pkg", "pkg.mod") in edges
    assert ("pkg.mod", "pkg.sub") in edges
    assert ("pkg.mod", "os") in edges
    assert ("main", "pkg") in edges

    assert graph["errors"]
    assert any("bad.py" in err["file"] for err in graph["errors"])

    excluded = python_dependency_graph(
        root=root,
        exclude_modules=["pkg.sub"],
    )
    assert "pkg.sub" not in excluded["nodes"]
    assert all(edge["to"] != "pkg.sub" for edge in excluded["edges"])

    truncated = python_dependency_graph(root=root, max_nodes=2)
    assert truncated["truncated"]["nodes"] is True


def test_code_quality_truncation_and_excludes(tmp_path):
    pytest.importorskip("prospector")
    pytest.importorskip("pyflakes")

    root = tmp_path / "proj"
    root.mkdir()
    _write(
        root / "app.py",
        "import os\nimport sys\n\ndef foo():\n    return bar\n",
    )
    _write(
        root / "tests" / "test_bad.py",
        "import math\n\ndef nope():\n    return missing\n",
    )

    result = code_quality(
        root=root,
        tools=["pyflakes"],
        exclude_folders=["tests"],
        max_messages=1,
    )

    assert result["message_count"] >= 2
    assert result["messages_returned"] == 1
    assert result["truncated"] is True
    assert all(
        not (msg["path"] or "").startswith("tests")
        for msg in result["messages"]
    )


def test_code_quality_multiple_directories(tmp_path):
    pytest.importorskip("prospector")
    pytest.importorskip("pyflakes")

    root = tmp_path / "repo"
    root.mkdir()
    _write(root / "one" / "a.py", "x = 1\n")
    _write(root / "two" / "b.py", "y = 2\n")

    result = code_quality(
        root=root,
        paths=["one", "two"],
        tools=["pyflakes"],
        max_messages=0,
    )

    assert result["files"] == 2
    assert len(result["runs"]) == 2


def test_python_dependency_graph_rejects_paths_outside_root(tmp_path):
    """Test that python_dependency_graph rejects paths outside the workspace root."""
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside / "test.py", "import os\n")

    with pytest.raises(ValueError, match="Path escapes workspace root"):
        python_dependency_graph(root=root, paths=[str(outside)])


def test_python_dependency_graph_relative_paths(tmp_path):
    """Test that python_dependency_graph works with relative paths."""
    root = tmp_path / "workspace"
    root.mkdir()
    _write(root / "test.py", "import os\n")

    graph = python_dependency_graph(root=root, paths=["test.py"])
    assert "test" in graph["nodes"]


def test_code_quality_rejects_paths_outside_root(tmp_path):
    """Test that code_quality rejects paths outside the workspace root."""
    pytest.importorskip("prospector")
    pytest.importorskip("pyflakes")

    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside / "test.py", "import os\n")

    with pytest.raises(ValueError, match="Path escapes workspace root"):
        code_quality(root=root, paths=[str(outside)], tools=["pyflakes"])


def test_code_quality_dotdot_paths(tmp_path):
    """Test that ../ in path is validated correctly."""
    pytest.importorskip("prospector")
    pytest.importorskip("pyflakes")

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sub").mkdir()

    with pytest.raises(ValueError, match="Path escapes workspace root"):
        code_quality(root=root / "sub", paths=["../outside.py"], tools=["pyflakes"])
