from codur.graph.tool_detection import create_default_tool_detector

def test_detect_json_tool_calls():
    detector = create_default_tool_detector()
    
    # Test case 1: JSON inside markdown code block
    message1 = """
    Here are the tool calls:
    ```json
    [
        {
            "tool": "read_file",
            "args": {"path": "test.py"}
        }
    ]
    ```
    """
    result1 = detector.detect(message1)
    assert result1 is not None
    # Should have read_file + python_ast_dependencies for .py file
    assert len(result1) == 2
    assert result1[0]["tool"] == "read_file"
    assert result1[0]["args"]["path"] == "test.py"
    assert result1[1]["tool"] == "python_ast_dependencies"
    assert result1[1]["args"]["path"] == "test.py"

    # Test case 2: Raw JSON array
    message2 = """
    [{"tool": "write_file", "args": {"path": "out.txt", "content": "hello"}}]
    """
    result2 = detector.detect(message2)
    assert result2 is not None
    assert len(result2) == 1
    assert result2[0]["tool"] == "write_file"
    assert result2[0]["args"]["content"] == "hello"

    # Test case 3: JSON object (not array) inside markdown
    message3 = """
    ```json
    {
        "tool": "list_files",
        "args": {"root": "."}
    }
    ```
    """
    result3 = detector.detect(message3)
    assert result3 is not None
    assert len(result3) == 1
    assert result3[0]["tool"] == "list_files"

    # Test case 4: Invalid JSON should be ignored
    message4 = """
    ```json
    { "tool": "broken" ... }
    ```
    """
    # Should fall back to other patterns or return empty list
    # In this case no patterns match
    result4 = detector.detect(message4)
    assert result4 == []


def test_detect_json_tool_calls_multiple_tools():
    detector = create_default_tool_detector()

    message = """
    ```json
    [
        {"tool": "read_file", "args": {"path": "a.py"}},
        {"tool": "list_files", "args": {"root": "."}}
    ]
    ```
    """
    result = detector.detect(message)
    assert result is not None
    # Should have read_file + list_files + python_ast_dependencies for .py file
    assert len(result) == 3
    assert result[0]["tool"] == "read_file"
    assert result[0]["args"]["path"] == "a.py"
    assert result[1]["tool"] == "list_files"
    assert result[2]["tool"] == "python_ast_dependencies"
    assert result[2]["args"]["path"] == "a.py"


def test_detect_json_tool_calls_skips_invalid_block():
    detector = create_default_tool_detector()

    message = """
    ```json
    { "tool": "broken" ... }
    ```
    ```json
    {"tool": "read_file", "args": {"path": "ok.py"}}
    ```
    """
    result = detector.detect(message)
    assert result is not None
    # Should have read_file + python_ast_dependencies for .py file
    assert len(result) == 2
    assert result[0]["tool"] == "read_file"
    assert result[0]["args"]["path"] == "ok.py"
    assert result[1]["tool"] == "python_ast_dependencies"
    assert result[1]["args"]["path"] == "ok.py"


def test_detect_json_tool_calls_uppercase_fence_ignored():
    detector = create_default_tool_detector()

    message = """
    ```JSON
    {"tool": "read_file", "args": {"path": "ok.py"}}
    ```
    """
    result = detector.detect(message)
    # Uppercase JSON fence is ignored, so no tools should be detected
    assert result == []


def test_detect_json_tool_calls_wrapped_object_ignored():
    detector = create_default_tool_detector()

    message = """
    ```json
    {"tool_calls": [{"tool": "read_file", "args": {"path": "ok.py"}}]}
    ```
    """
    result = detector.detect(message)
    # Wrapped in tool_calls object instead of direct array, so no tools should be detected
    assert result == []
