import os
import shutil
from pathlib import Path
import pytest
from codur.tools.filesystem import (
    read_file, write_file, append_file, delete_file,
    copy_file, move_file, list_files, search_files,
    replace_in_file, line_count, inject_lines, replace_lines
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
    files = list_files(root=temp_fs)
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
