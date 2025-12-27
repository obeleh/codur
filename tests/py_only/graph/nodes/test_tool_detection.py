from codur.graph.nodes.tool_detection import create_default_tool_detector


def test_change_intent_reads_file() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("Fix bug in @app.py")
    # Should detect read_file + python_ast_dependencies for .py file
    assert result == [
        {"tool": "read_file", "args": {"path": "app.py"}},
        {"tool": "python_ast_dependencies", "args": {"path": "app.py"}}
    ]


def test_move_file_to_dir_trailing_slash() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("move src/main.py to backup/")
    assert result == [{"tool": "move_file_to_dir", "args": {"source": "src/main.py", "destination_dir": "backup"}}]


def test_set_ini_value() -> None:
    detector = create_default_tool_detector()
    result = detector.detect('set ini core.timeout in config.ini to "30"')
    assert result == [
        {"tool": "set_ini_value", "args": {"path": "config.ini", "section": "core", "option": "timeout", "value": "30"}}
    ]


def test_lint_python_tree() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("lint python tree src")
    assert result == [{"tool": "lint_python_tree", "args": {"root": "src"}}]


def test_read_file_requires_path_like_token() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("read the file")
    # Should return empty list when no valid paths are found
    assert result == []


def test_change_intent_ignores_non_path_after_in() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("fix the bug in the code")
    # Should return empty list when "in" is followed by non-path tokens
    assert result == []


def test_write_file_rejects_task_description() -> None:
    """Test that 'Write unit tests' task description is not interpreted as file write."""
    detector = create_default_tool_detector()
    result = detector.detect(
        "Implement the is_palindrome function in @main.py based on the docstring. "
        "Write unit tests in @test_main.py. Run pytest to validate all requirements."
    )
    # Should not detect a write_file to "validate" - the path "validate" is not valid
    assert result is None or not any(
        tool.get("tool") == "write_file" and tool.get("args", {}).get("path") == "validate"
        for tool in result
    )


def test_write_file_accepts_valid_file_path() -> None:
    """Test that write_file still works with valid file paths."""
    detector = create_default_tool_detector()
    result = detector.detect('write "hello world" to test.txt')
    assert result == [{"tool": "write_file", "args": {"path": "test.txt", "content": "hello world"}}]


def test_markdown_file_injection() -> None:
    """Test that markdown files get markdown_outline injected."""
    detector = create_default_tool_detector()
    result = detector.detect("read README.md")
    # Should detect read_file + markdown_outline for .md file
    assert result == [
        {"tool": "read_file", "args": {"path": "README.md"}},
        {"tool": "markdown_outline", "args": {"path": "README.md"}}
    ]


def test_markdown_file_with_fix_intent() -> None:
    """Test that fix intent with markdown file triggers injection."""
    detector = create_default_tool_detector()
    result = detector.detect("Fix the table in @CONTRIBUTING.md")
    # Should detect read_file + markdown_outline for .md file
    assert result == [
        {"tool": "read_file", "args": {"path": "CONTRIBUTING.md"}},
        {"tool": "markdown_outline", "args": {"path": "CONTRIBUTING.md"}}
    ]


def test_rename_symbol_detection() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("rename foo to bar in @app.py")
    assert result == [
        {"tool": "rope_rename_symbol", "args": {"path": "app.py", "symbol": "foo", "new_name": "bar"}}
    ]


def test_json_tool_calls_with_python_file() -> None:
    """Test JSON tool calls format with Python file injection."""
    detector = create_default_tool_detector()
    result = detector.detect('''
    ```json
    [{"tool": "read_file", "args": {"path": "utils.py"}}]
    ```
    ''')
    # Should parse JSON and inject Python AST
    assert len(result) == 2
    assert result[0] == {"tool": "read_file", "args": {"path": "utils.py"}}
    assert result[1]["tool"] == "python_ast_dependencies"
    assert result[1]["args"]["path"] == "utils.py"


def test_json_tool_calls_with_markdown_file() -> None:
    """Test JSON tool calls format with Markdown file injection."""
    detector = create_default_tool_detector()
    result = detector.detect('''
    ```json
    [{"tool": "read_file", "args": {"path": "CHANGELOG.md"}}]
    ```
    ''')
    # Should parse JSON and inject Markdown outline
    assert len(result) == 2
    assert result[0] == {"tool": "read_file", "args": {"path": "CHANGELOG.md"}}
    assert result[1]["tool"] == "markdown_outline"
    assert result[1]["args"]["path"] == "CHANGELOG.md"
