import pytest
from codur.graph.nodes.tool_detection import create_default_tool_detector

def test_detect_json_tool_calls():
    detector = create_default_tool_detector()
    
    # Test case 1: JSON inside markdown code block
    message1 = """
    I will read the file now.
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
    assert len(result1) == 1
    assert result1[0]["tool"] == "read_file"
    assert result1[0]["args"]["path"] == "test.py"

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
    # Should fall back to other patterns or return None
    # In this case likely None unless it matches textual patterns
    result4 = detector.detect(message4)
    assert result4 is None


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
    assert len(result) == 2
    assert result[0]["tool"] == "read_file"
    assert result[1]["tool"] == "list_files"


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
    assert result[0]["tool"] == "read_file"


def test_detect_json_tool_calls_uppercase_fence_ignored():
    detector = create_default_tool_detector()

    message = """
    ```JSON
    {"tool": "read_file", "args": {"path": "ok.py"}}
    ```
    """
    result = detector.detect(message)
    assert result is None


def test_detect_json_tool_calls_wrapped_object_ignored():
    detector = create_default_tool_detector()

    message = """
    ```json
    {"tool_calls": [{"tool": "read_file", "args": {"path": "ok.py"}}]}
    ```
    """
    result = detector.detect(message)
    assert result is None
