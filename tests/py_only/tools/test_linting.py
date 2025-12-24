"""Tests for linting tools."""

from pathlib import Path

import pytest

from codur.tools.linting import lint_python_files, lint_python_tree


def test_lint_python_files_handles_missing_and_syntax_error(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    result = lint_python_files(
        paths=["bad.py", "missing.py"],
        root=root,
        max_errors=10,
    )

    assert result["checked"] == 2
    assert len(result["errors"]) == 2
    assert any("Failed to read file" in err["message"] for err in result["errors"])
    assert any(err["file"].endswith("bad.py") for err in result["errors"])

    limited = lint_python_files(
        paths=["bad.py", "missing.py"],
        root=root,
        max_errors=1,
    )
    assert limited["checked"] == 1
    assert len(limited["errors"]) == 1


def test_lint_python_files_rejects_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    external = tmp_path / "outside.py"
    external.write_text("print('ok')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Path escapes workspace root"):
        lint_python_files([str(external)], root=root)


def test_lint_python_tree_skips_excluded_dirs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "good.py").write_text("print('ok')\n", encoding="utf-8")
    venv = root / ".venv"
    venv.mkdir()
    (venv / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    result = lint_python_tree(root=root)
    assert result["checked"] == 1
    assert result["errors"] == []
