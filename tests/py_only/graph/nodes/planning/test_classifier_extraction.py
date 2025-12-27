"""Tests for classifier file extraction."""

from codur.graph.nodes.planning.classifier import extract_file_paths

def test_extract_file_paths_basic():
    text = "Fix the bug in main.py"
    paths = extract_file_paths(text)
    assert "main.py" in paths
    assert len(paths) == 1

def test_extract_file_paths_with_at_symbol():
    text = "Fix the bug in @main.py"
    paths = extract_file_paths(text)
    assert "main.py" in paths
    assert "@main.py" not in paths
    assert len(paths) == 1

def test_extract_file_paths_mixed():
    text = "Check @main.py and utils.py"
    paths = extract_file_paths(text)
    assert "main.py" in paths
    assert "utils.py" in paths
    assert len(paths) == 2

def test_extract_file_paths_quoted():
    text = "Edit 'path/to/file.py'"
    paths = extract_file_paths(text)
    assert "path/to/file.py" in paths

def test_extract_file_paths_duplicates():
    text = "Fix main.py and @main.py"
    paths = extract_file_paths(text)
    assert paths == ["main.py"]

def test_extract_file_paths_extensions():
    # Test various extensions
    text = "config.yaml, data.json, script.ts"
    paths = extract_file_paths(text)
    assert "config.yaml" in paths
    assert "data.json" in paths
    assert "script.ts" in paths

def test_extract_file_paths_backticks():
    text = "Implement the `format_table` function in `main.py` for the formatter."
    paths = extract_file_paths(text)
    assert "main.py" in paths
    assert len(paths) == 1

def test_extract_file_paths_backticks_with_command():
    text = (
        "Implement the `format_table` function in `main.py` to create a working "
        "Markdown table formatter. Run `python main.py` to test your solution."
    )
    paths = extract_file_paths(text)
    assert "main.py" in paths
    assert "python main.py" not in paths
    assert len(paths) == 1
