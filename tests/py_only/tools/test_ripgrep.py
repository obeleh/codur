import os
from pathlib import Path

import pytest

from codur.tools.ripgrep import grep_files, ripgrep_search


@pytest.fixture
def temp_fs(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "caps.txt").write_text("Hello World", encoding="utf-8")
    (root / "lower.txt").write_text("hello world", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00hello")
    (root / "sub").mkdir()
    (root / "sub" / "match.txt").write_text("hello sub", encoding="utf-8")
    return root


def test_grep_files_case_sensitive_and_binary(temp_fs):
    insensitive = grep_files("hello", root=temp_fs)
    files = {entry["file"] for entry in insensitive}
    assert "caps.txt" in files
    assert "lower.txt" in files
    assert "binary.bin" not in files
    sensitive = grep_files("hello", root=temp_fs, case_sensitive=True)
    files = {entry["file"] for entry in sensitive}
    assert "lower.txt" in files
    assert "caps.txt" not in files


def test_ripgrep_search_glob_filters(temp_fs):
    results = ripgrep_search("hello", root=temp_fs, globs=["sub/*.txt"])
    files = {entry["file"] for entry in results}
    assert os.path.join("sub", "match.txt") in files
    assert "lower.txt" not in files


def test_ripgrep_search_fixed_strings(temp_fs):
    target = Path(temp_fs) / "regex.txt"
    target.write_text("abc", encoding="utf-8")
    results = ripgrep_search("a.c", root=temp_fs, fixed_strings=True)
    assert results == []


def test_ripgrep_search_invalid_regex_raises(temp_fs):
    with pytest.raises(ValueError, match="ripgrep error"):
        ripgrep_search("(", root=temp_fs)


def test_ripgrep_search_max_results(temp_fs):
    results = ripgrep_search("hello", root=temp_fs, max_results=1)
    assert len(results) == 1


def test_ripgrep_search_hidden_flag_controls_hidden_files(temp_fs):
    hidden_dir = Path(temp_fs) / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("hello hidden", encoding="utf-8")
    results = ripgrep_search("hello", root=temp_fs)
    files = {entry["file"] for entry in results}
    assert os.path.join(".hidden", "secret.txt") not in files
    results_hidden = ripgrep_search("hello", root=temp_fs, hidden=True)
    files_hidden = {entry["file"] for entry in results_hidden}
    assert os.path.join(".hidden", "secret.txt") in files_hidden


def test_grep_files_excludes_git_dir(temp_fs):
    git_dir = Path(temp_fs) / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("hello git", encoding="utf-8")
    results = grep_files("hello", root=temp_fs)
    files = {entry["file"] for entry in results}
    assert os.path.join(".git", "config") not in files


def test_ripgrep_search_returns_relative_paths(temp_fs):
    results = ripgrep_search("hello", root=temp_fs, max_results=1)
    assert results
    assert not os.path.isabs(results[0]["file"])


def test_ripgrep_search_negated_glob_excludes_matches(temp_fs):
    results = ripgrep_search("hello", root=temp_fs, globs=["*.txt", "!lower.txt"])
    files = {entry["file"] for entry in results}
    assert "lower.txt" not in files


def test_grep_files_falls_back_when_rg_missing(temp_fs, monkeypatch):
    monkeypatch.setattr("codur.tools.ripgrep._rg_available", lambda: False)
    results = grep_files("hello", root=temp_fs)
    files = {entry["file"] for entry in results}
    assert "lower.txt" in files
    assert "caps.txt" in files


def test_ripgrep_search_errors_when_rg_missing(temp_fs, monkeypatch):
    monkeypatch.setattr("codur.tools.ripgrep._rg_available", lambda: False)
    with pytest.raises(ValueError, match="ripgrep not found"):
        ripgrep_search("hello", root=temp_fs)


# Path resolution tests for grep_files
def test_grep_files_with_path_parameter(temp_fs):
    """Test that grep_files correctly uses the path parameter."""
    results = grep_files("hello", path="sub", root=temp_fs)
    files = {entry["file"] for entry in results}
    # Should only find matches in the sub directory
    assert "match.txt" in files
    assert "lower.txt" not in files


def test_grep_files_path_and_root_interaction(temp_fs):
    """Test that path parameter is resolved relative to root."""
    # Create nested structure
    nested = temp_fs / "nested"
    nested.mkdir()
    (nested / "test.txt").write_text("hello nested", encoding="utf-8")

    # Search with path relative to root
    results = grep_files("hello", path="nested", root=temp_fs)
    files = {entry["file"] for entry in results}
    assert "test.txt" in files


def test_grep_files_rejects_path_outside_root(temp_fs):
    """Test that grep_files rejects paths that escape the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        grep_files("hello", path="../outside", root=temp_fs)


def test_grep_files_absolute_path_inside_root(temp_fs):
    """Test that grep_files accepts absolute paths within the root."""
    subdir = temp_fs / "sub"
    results = grep_files("hello", path=str(subdir), root=temp_fs)
    files = {entry["file"] for entry in results}
    assert "match.txt" in files


def test_grep_files_absolute_path_outside_root(temp_fs):
    """Test that grep_files rejects absolute paths outside the root."""
    outside = temp_fs.parent / "outside"
    outside.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        grep_files("hello", path=str(outside), root=temp_fs)


def test_grep_files_path_without_root(temp_fs):
    """Test that when path is provided without root, it searches that path."""
    # When path is an absolute path within the workspace
    subdir = temp_fs / "sub"
    from codur.utils.path_utils import set_default_root

    # Set the default root so the path validation works
    old_root = None
    set_default_root(temp_fs)
    try:
        results = grep_files("hello", path="sub")
        files = {entry["file"] for entry in results}
        assert "match.txt" in files
    finally:
        set_default_root(old_root)


def test_grep_files_with_dotdot_path(temp_fs):
    """Test that ../ in path is resolved correctly and validated."""
    # Create structure
    (temp_fs / "a").mkdir()
    (temp_fs / "b").mkdir()
    (temp_fs / "b" / "test.txt").write_text("hello", encoding="utf-8")

    # Try to access sibling directory using ../
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        grep_files("hello", path="../b", root=temp_fs / "a")
