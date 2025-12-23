from codur.graph.nodes.tool_detection import create_default_tool_detector


def test_change_intent_reads_file() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("Fix bug in @app.py")
    assert result == [{"tool": "read_file", "args": {"path": "app.py"}}]


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
    assert result is None


def test_change_intent_ignores_non_path_after_in() -> None:
    detector = create_default_tool_detector()
    result = detector.detect("fix the bug in the code")
    assert result is None
