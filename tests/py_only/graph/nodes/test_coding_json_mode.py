"""
Tests for JSON-mode tool calling (fallback when native tools not supported).
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from codur.utils.tool_response_handler import extract_tool_calls_from_json_text


class TestCodingJsonMode:
    """Test the JSON parsing and tool dispatching of the coding node."""

    def test_extract_valid_json_replace_function(self):
        """Test extracting tool calls from valid JSON response."""
        new_code = "def old_func():\n    return 'new'"
        json_payload = {
            "thought": "I will replace the function.",
            "tool_calls": [
                {
                    "tool": "replace_function",
                    "args": {
                        "path": "main.py",
                        "function_name": "old_func",
                        "new_code": new_code,
                    },
                }
            ],
        }
        response_text = f"```json\n{json.dumps(json_payload)}\n```"
        message = AIMessage(content=response_text)

        tool_calls = extract_tool_calls_from_json_text(message)

        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "replace_function"
        assert tool_calls[0]["args"]["path"] == "main.py"
        assert tool_calls[0]["args"]["function_name"] == "old_func"
        assert "return 'new'" in tool_calls[0]["args"]["new_code"]

    def test_extract_invalid_json_returns_empty(self):
        """Test that malformed JSON returns empty list."""
        response_text = '{"thought": "oops", "tool_calls": [}'
        message = AIMessage(content=response_text)

        tool_calls = extract_tool_calls_from_json_text(message)
        assert tool_calls == []

    def test_extract_json_with_code_block(self):
        """Test that JSON wrapped in markdown code block is extracted."""
        json_payload = {
            "tool_calls": [
                {
                    "tool": "read_file",
                    "args": {"path": "main.py"},
                }
            ],
        }
        response_text = f"Here you go:\n```json\n{json.dumps(json_payload)}\n```\nThanks!"
        message = AIMessage(content=response_text)

        tool_calls = extract_tool_calls_from_json_text(message)

        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "read_file"
        assert tool_calls[0]["args"]["path"] == "main.py"

    def test_extract_multiple_tool_calls(self):
        """Test extracting multiple tool calls from JSON."""
        json_payload = {
            "tool_calls": [
                {
                    "tool": "read_file",
                    "args": {"path": "file1.py"},
                },
                {
                    "tool": "read_file",
                    "args": {"path": "file2.py"},
                },
            ],
        }
        response_text = json.dumps(json_payload)
        message = AIMessage(content=response_text)

        tool_calls = extract_tool_calls_from_json_text(message)

        assert len(tool_calls) == 2
        assert tool_calls[0]["tool"] == "read_file"
        assert tool_calls[0]["args"]["path"] == "file1.py"
        assert tool_calls[1]["tool"] == "read_file"
        assert tool_calls[1]["args"]["path"] == "file2.py"

    def test_extract_empty_tool_calls_returns_empty(self):
        """Test that empty tool_calls list returns empty list."""
        json_payload = {"thought": "Nothing to do.", "tool_calls": []}
        response_text = json.dumps(json_payload)
        message = AIMessage(content=response_text)

        tool_calls = extract_tool_calls_from_json_text(message)

        assert tool_calls == []
