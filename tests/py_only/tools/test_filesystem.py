import os
import shutil
from pathlib import Path
from types import SimpleNamespace
import pytest
from codur.tools.filesystem import (
    read_file, write_file, append_file, delete_file,
    copy_file, move_file, list_files, search_files,
    replace_in_file, line_count, inject_lines, replace_lines,
    list_dirs, file_tree, copy_file_to_dir
)

@pytest.fixture
def temp_fs(tmp_path):
    """Create a temporary filesystem structure for testing."""
    root = tmp_path / "workspace"
    root.mkdir()
    
    # Create some files
    (root / "file1.txt").write_text("Hello World", encoding="utf-8")
    (root / "file2.txt").write_text("Hello Python", encoding="utf-8")
    (root / "subdir").mkdir()
    (root / "subdir" / "file3.txt").write_text("Nested file", encoding="utf-8")
    
    return root

def test_read_file(temp_fs):
    content = read_file("file1.txt", root=temp_fs)
    assert content == "Hello World"

def test_write_file(temp_fs):
    write_file("new.txt", "New Content", root=temp_fs)
    assert (temp_fs / "new.txt").read_text(encoding="utf-8") == "New Content"

def test_append_file(temp_fs):
    append_file("file1.txt", "\nAppended", root=temp_fs)
    content = (temp_fs / "file1.txt").read_text(encoding="utf-8")
    assert content == "Hello World\nAppended"

def test_delete_file(temp_fs):
    delete_file("file1.txt", root=temp_fs)
    assert not (temp_fs / "file1.txt").exists()

def test_copy_file(temp_fs):
    copy_file("file1.txt", "copy.txt", root=temp_fs)
    assert (temp_fs / "copy.txt").exists()
    assert (temp_fs / "copy.txt").read_text(encoding="utf-8") == "Hello World"

def test_move_file(temp_fs):
    move_file("file1.txt", "moved.txt", root=temp_fs)
    assert not (temp_fs / "file1.txt").exists()
    assert (temp_fs / "moved.txt").exists()
    assert (temp_fs / "moved.txt").read_text(encoding="utf-8") == "Hello World"

def test_list_files(temp_fs):
    files = list_files(path=temp_fs)
    assert "file1.txt" in files
    assert "file2.txt" in files
    assert os.path.join("subdir", "file3.txt") in files

def test_search_files(temp_fs):
    results = search_files("file", root=temp_fs)
    assert len(results) >= 3
    
    # Test subdirectory search behavior if applicable, 
    # though search_files just matches filename substrings based on implementation
    
def test_replace_in_file(temp_fs):
    replace_in_file("file1.txt", "World", "Universe", root=temp_fs)
    content = (temp_fs / "file1.txt").read_text(encoding="utf-8")
    assert content == "Hello Universe"

def test_line_count(temp_fs):
    write_file("lines.txt", "Line 1\nLine 2\nLine 3", root=temp_fs)
    result = line_count("lines.txt", root=temp_fs)
    assert result["lines"] == 3


def test_inject_lines_at_start(temp_fs):
    write_file("lines.txt", "Line 1\nLine 2\n", root=temp_fs)
    inject_lines("lines.txt", line=1, content="Start\n", root=temp_fs)
    content = (temp_fs / "lines.txt").read_text(encoding="utf-8")
    assert content == "Start\nLine 1\nLine 2\n"


def test_inject_lines_in_middle(temp_fs):
    write_file("lines.txt", "Line 1\nLine 2\nLine 3\n", root=temp_fs)
    inject_lines("lines.txt", line=2, content="Inserted\n", root=temp_fs)
    content = (temp_fs / "lines.txt").read_text(encoding="utf-8")
    assert content == "Line 1\nInserted\nLine 2\nLine 3\n"


def test_inject_lines_at_end(temp_fs):
    write_file("lines.txt", "Line 1\nLine 2\n", root=temp_fs)
    inject_lines("lines.txt", line=3, content="End\n", root=temp_fs)
    content = (temp_fs / "lines.txt").read_text(encoding="utf-8")
    assert content == "Line 1\nLine 2\nEnd\n"


def test_replace_lines_range(temp_fs):
    write_file("lines.txt", "Line 1\nLine 2\nLine 3\nLine 4\n", root=temp_fs)
    replace_lines("lines.txt", start_line=2, end_line=3, content="New A\nNew B\n", root=temp_fs)
    content = (temp_fs / "lines.txt").read_text(encoding="utf-8")
    assert content == "Line 1\nNew A\nNew B\nLine 4\n"


def test_replace_lines_append(temp_fs):
    write_file("lines.txt", "Line 1\n", root=temp_fs)
    replace_lines("lines.txt", start_line=2, end_line=2, content="Line 2\n", root=temp_fs)
    content = (temp_fs / "lines.txt").read_text(encoding="utf-8")
    assert content == "Line 1\nLine 2\n"


def test_inject_lines_invalid_line(temp_fs):
    write_file("lines.txt", "Line 1\n", root=temp_fs)
    with pytest.raises(ValueError):
        inject_lines("lines.txt", line=0, content="Nope\n", root=temp_fs)


def test_replace_lines_invalid_range(temp_fs):
    write_file("lines.txt", "Line 1\n", root=temp_fs)
    with pytest.raises(ValueError):
        replace_lines("lines.txt", start_line=3, end_line=2, content="Nope\n", root=temp_fs)


def test_replace_in_file_no_matches(temp_fs):
    write_file("no_match.txt", "alpha beta", root=temp_fs)
    result = replace_in_file("no_match.txt", "gamma", "delta", root=temp_fs)
    assert result["replacements"] == 0
    assert (temp_fs / "no_match.txt").read_text(encoding="utf-8") == "alpha beta"


def test_inject_lines_line_too_large(temp_fs):
    write_file("lines.txt", "Line 1\n", root=temp_fs)
    with pytest.raises(ValueError, match="line exceeds file length"):
        inject_lines("lines.txt", line=3, content="Nope\n", root=temp_fs)


def test_replace_lines_end_line_too_large(temp_fs):
    write_file("lines.txt", "Line 1\n", root=temp_fs)
    with pytest.raises(ValueError, match="end_line exceeds file length"):
        replace_lines("lines.txt", start_line=1, end_line=3, content="Nope\n", root=temp_fs)


def test_line_count_empty_file(temp_fs):
    write_file("empty.txt", "", root=temp_fs)
    result = line_count("empty.txt", root=temp_fs)
    assert result["lines"] == 0


def test_read_file_rejects_outside_root(temp_fs):
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        read_file("../outside.txt", root=temp_fs)


def test_read_file_blocks_secret_by_default(temp_fs):
    secret_path = temp_fs / ".env"
    secret_path.write_text("SECRET=1", encoding="utf-8")
    config = SimpleNamespace(tools=SimpleNamespace(allow_read_secrets=False, secret_globs=[]))
    state = {"config": config}
    with pytest.raises(ValueError, match="secret files"):
        read_file(".env", root=temp_fs, state=state)


def test_read_file_allows_secret_when_enabled(temp_fs):
    secret_path = temp_fs / ".env"
    secret_path.write_text("SECRET=1", encoding="utf-8")
    config = SimpleNamespace(tools=SimpleNamespace(allow_read_secrets=True, secret_globs=[]))
    state = {"config": config}
    content = read_file(".env", root=temp_fs, state=state)
    assert "SECRET=1" in content


def test_list_dirs_excludes_special_and_limits(temp_fs):
    (temp_fs / ".git").mkdir()
    (temp_fs / "dirA").mkdir()
    (temp_fs / "dirA" / "child").mkdir()
    dirs = list_dirs(root=temp_fs)
    assert ".git" not in dirs
    assert "dirA" in dirs
    assert os.path.join("dirA", "child") in dirs
    limited = list_dirs(root=temp_fs, max_results=1)
    assert len(limited) == 1


def test_file_tree_file_path_returns_absolute(temp_fs):
    target = temp_fs / "file1.txt"
    results = file_tree(path="file1.txt", root=temp_fs)
    assert len(results) == 1
    assert Path(results[0]).is_absolute()
    assert results[0].endswith(str(target))


def test_file_tree_max_depth(temp_fs):
    (temp_fs / "deep" / "nested").mkdir(parents=True)
    (temp_fs / "deep" / "nested" / "file.txt").write_text("hi", encoding="utf-8")
    results = file_tree(root=temp_fs, max_depth=0)
    assert "deep/" in results
    assert "deep/nested/" not in results


def test_copy_file_to_dir_create_dirs_false_raises(temp_fs):
    with pytest.raises(OSError):
        copy_file_to_dir("file1.txt", "missing", root=temp_fs, create_dirs=False)
    result = copy_file_to_dir("file1.txt", "created", root=temp_fs, create_dirs=True)
    assert "created" in result
    assert (temp_fs / "created" / "file1.txt").exists()
