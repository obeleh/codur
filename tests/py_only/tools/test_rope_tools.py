"""Tests for rope-based tools."""

from pathlib import Path
from tempfile import TemporaryDirectory

from codur.tools.rope_tools import (
    rope_find_usages,
    rope_find_definition,
    rope_rename_symbol,
    rope_move_module,
    rope_extract_method,
)


def test_rope_find_usages_across_files():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def foo():\n    return 1\n\nfoo()\n")
        (root / "b.py").write_text("from a import foo\n\nfoo()\n")

        result = rope_find_usages(path="a.py", line=1, column=4, root=root)

        paths = {item["path"] for item in result["occurrences"]}
        assert "a.py" in paths
        assert "b.py" in paths
        assert result["count"] == len(result["occurrences"])
        assert result["truncated"] is False


def test_rope_find_definition():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def foo():\n    return 1\n\nfoo()\n")

        result = rope_find_definition(path="a.py", line=4, column=0, root=root)

        assert result["found"] is True
        definition = result["definition"]
        assert definition["path"] == "a.py"
        assert definition["line"] == 1


def test_rope_rename_symbol_updates_references():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def foo():\n    return 1\n\nfoo()\n")
        (root / "b.py").write_text("from a import foo\n\nfoo()\n")

        result = rope_rename_symbol(path="a.py", line=1, column=4, new_name="bar", root=root)

        assert "a.py" in result["changed_files"]
        assert "b.py" in result["changed_files"]
        assert "foo" not in (root / "a.py").read_text()
        assert "foo" not in (root / "b.py").read_text()
        assert "bar" in (root / "a.py").read_text()
        assert "bar" in (root / "b.py").read_text()


def test_rope_rename_symbol_by_name():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def foo():\n    return 1\n\nfoo()\n")

        result = rope_rename_symbol(path="a.py", symbol="foo", new_name="bar", root=root)

        assert result["symbol"] == "foo"
        assert "foo" not in (root / "a.py").read_text()
        assert "bar" in (root / "a.py").read_text()


def test_rope_move_module_moves_file():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def foo():\n    return 1\n")
        (root / "b.py").write_text("import a\n\nprint(a.foo())\n")
        (root / "pkg").mkdir()

        result = rope_move_module(path="a.py", destination_dir="pkg", root=root)

        assert "pkg/a.py" in result["changed_files"]
        assert not (root / "a.py").exists()
        assert (root / "pkg" / "a.py").exists()
        assert "import pkg.a" in (root / "b.py").read_text()


def test_rope_extract_method_creates_new_method():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        content = (
            "def foo():\n"
            "    x = 1\n"
            "    y = 2\n"
            "    return x + y\n"
        )
        path = root / "a.py"
        path.write_text(content)
        start_offset = content.index("x = 1")
        end_offset = content.index("return")

        result = rope_extract_method(
            path="a.py",
            extracted_name="calc",
            start_offset=start_offset,
            end_offset=end_offset,
            root=root,
        )

        updated = path.read_text()
        assert "def calc()" in updated
        assert "x = 1" in updated
        assert "return x, y" in updated
        assert "changed_files" in result


def test_rope_find_usages_rejects_outside_root():
    """Test that rope_find_usages rejects paths outside the workspace root."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "workspace"
        root.mkdir()
        (root / "a.py").write_text("def foo():\n    return 1\n")

        outside = Path(tmpdir) / "outside.py"
        outside.write_text("def bar():\n    return 2\n")

        import pytest
        with pytest.raises(ValueError, match="Path escapes workspace root"):
            rope_find_usages(path=str(outside), line=1, column=4, root=root)


def test_rope_find_definition_rejects_outside_root():
    """Test that rope_find_definition rejects paths outside the workspace root."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "workspace"
        root.mkdir()
        (root / "a.py").write_text("def foo():\n    return 1\n")

        outside = Path(tmpdir) / "outside.py"
        outside.write_text("def bar():\n    return 2\n")

        import pytest
        with pytest.raises(ValueError, match="Path escapes workspace root"):
            rope_find_definition(path=str(outside), line=1, column=4, root=root)


def test_rope_rename_symbol_rejects_outside_root():
    """Test that rope_rename_symbol rejects paths outside the workspace root."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "workspace"
        root.mkdir()
        (root / "a.py").write_text("def foo():\n    return 1\n")

        outside = Path(tmpdir) / "outside.py"
        outside.write_text("def bar():\n    return 2\n")

        import pytest
        with pytest.raises(ValueError, match="Path escapes workspace root"):
            rope_rename_symbol(path=str(outside), line=1, column=4, new_name="baz", root=root)


def test_rope_move_module_rejects_outside_root():
    """Test that rope_move_module rejects paths outside the workspace root."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "workspace"
        root.mkdir()
        (root / "a.py").write_text("def foo():\n    return 1\n")
        (root / "pkg").mkdir()

        outside = Path(tmpdir) / "outside.py"
        outside.write_text("def bar():\n    return 2\n")

        import pytest
        with pytest.raises(ValueError, match="Path escapes workspace root"):
            rope_move_module(path=str(outside), destination_dir="pkg", root=root)


def test_rope_extract_method_rejects_outside_root():
    """Test that rope_extract_method rejects paths outside the workspace root."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "workspace"
        root.mkdir()

        outside = Path(tmpdir) / "outside.py"
        content = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        outside.write_text(content)

        import pytest
        with pytest.raises(ValueError, match="Path escapes workspace root"):
            rope_extract_method(
                path=str(outside),
                extracted_name="calc",
                start_offset=0,
                end_offset=10,
                root=root,
            )


def test_rope_find_usages_relative_path():
    """Test that rope_find_usages works with relative paths."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def foo():\n    return 1\n\nfoo()\n")

        result = rope_find_usages(path="a.py", line=1, column=4, root=root)
        assert result["count"] > 0
