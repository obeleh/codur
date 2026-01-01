"""Tests for tool analysis logic."""

import json

from langchain_core.messages import ToolMessage

from codur.graph.planning.tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
    extract_list_files,
    pick_preferred_python_file,
)


def _tool_msg(tool: str, output, args: dict | None = None) -> ToolMessage:
    """Helper to create a ToolMessage in the new JSON format."""
    return ToolMessage(
        content=json.dumps({"tool": tool, "output": output, "args": args or {}}),
        tool_call_id="test",
    )


class TestToolAnalysis:
    def test_tool_results_include_read_file(self):
        messages = [_tool_msg("read_file", "content of file", {"path": "main.py"})]
        assert tool_results_include_read_file(messages) is True

        messages = [_tool_msg("list_files", ["a.py"])]
        assert tool_results_include_read_file(messages) is False

    def test_extract_list_files(self):
        messages = [_tool_msg("list_files", ["main.py", "test.py"])]
        files = extract_list_files(messages)
        assert files == ["main.py", "test.py"]

        messages = [_tool_msg("grep_files", "some other output")]
        files = extract_list_files(messages)
        assert files == []

    def test_pick_preferred_python_file(self):
        # Case 1: Prefers main.py
        files = ["utils.py", "main.py", "test.py"]
        assert pick_preferred_python_file(files) == "main.py"

        # Case 2: Prefers app.py
        files = ["utils.py", "app.py"]
        assert pick_preferred_python_file(files) == "app.py"

        # Case 3: Prefers shallowest path
        files = ["src/utils.py", "root_script.py"]
        assert pick_preferred_python_file(files) == "root_script.py"

        # Case 4: No py files
        files = ["README.md", "config.json"]
        assert pick_preferred_python_file(files) is None

    def test_select_file_from_tool_results(self):
        messages = [_tool_msg("list_files", ["utils.py", "main.py"])]
        selected = select_file_from_tool_results(messages)
        assert selected == "main.py"
