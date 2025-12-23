"""
Tests for the refactored, JSON-mode coding node.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
from codur.graph.nodes.coding import _apply_coding_result


class TestCodingJsonMode:
    """Test the JSON parsing and tool dispatching of the coding node."""

    def test_apply_valid_json_replace_function(self):
        """Test applying a valid JSON response for a function replacement."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text("def old_func():\n    pass")

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
            response = f"```json\n{json.dumps(json_payload)}\n```"

            config = MagicMock()
            config.runtime.allow_outside_workspace = False
            state = {}

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                error = _apply_coding_result(response, state, config)
                assert error is None, f"Unexpected error: {error}"

                content = file_path.read_text()
                assert "return 'new'" in content
            finally:
                os.chdir(old_cwd)

    def test_apply_invalid_json_fails(self):
        """Test that malformed JSON returns an error."""
        response = '{"thought": "oops", "tool_calls": [}'
        config = MagicMock()
        state = {}

        error = _apply_coding_result(response, state, config)
        assert error is not None
        assert "Failed to parse JSON" in error

    def test_apply_unknown_tool_fails(self):
        """Test that a response with an unknown tool name returns an error."""
        json_payload = {
            "thought": "Let's try a fake tool.",
            "tool_calls": [{"tool": "invent_tool", "args": {}}],
        }
        response = json.dumps(json_payload)
        config = MagicMock()
        state = {}

        error = _apply_coding_result(response, state, config)
        assert error is not None
        assert "Unknown tool: invent_tool" in error

    def test_apply_tool_with_invalid_syntax_fails(self):
        """Test that a tool call with invalid python code fails validation."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text("def old_func():\n    pass")

            invalid_code = "def old_func():\n  return 'oops"
            json_payload = {
                "thought": "This code has a syntax error.",
                "tool_calls": [
                    {
                        "tool": "replace_function",
                        "args": {
                            "path": "main.py",
                            "function_name": "old_func",
                            "new_code": invalid_code,
                        },
                    }
                ],
            }
            response = json.dumps(json_payload)

            config = MagicMock()
            config.runtime.allow_outside_workspace = False
            state = {}
            
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                error = _apply_coding_result(response, state, config)
                assert error is not None
                assert "Invalid Python syntax" in error
                assert invalid_code in error
            finally:
                os.chdir(old_cwd)

    def test_apply_json_with_prose_code_block(self):
        """Test that JSON wrapped in a code block with prose still parses."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text("def foo():\n    return 1\n")

            new_code = "def foo():\n    return 2"
            json_payload = {
                "thought": "Replace foo.",
                "tool_calls": [
                    {
                        "tool": "replace_function",
                        "args": {
                            "path": "main.py",
                            "function_name": "foo",
                            "new_code": new_code,
                        },
                    }
                ],
            }
            response = f"Here you go:\n```json\n{json.dumps(json_payload)}\n```\nThanks!"

            config = MagicMock()
            config.runtime.allow_outside_workspace = False
            state = {}

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                error = _apply_coding_result(response, state, config)
                assert error is None
                assert "return 2" in file_path.read_text()
            finally:
                os.chdir(old_cwd)

    def test_apply_json_missing_path_defaults_to_main(self):
        """Test that missing path defaults to main.py."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text("def foo():\n    return 1\n")

            new_code = "def foo():\n    return 3"
            json_payload = {
                "thought": "Replace foo without path.",
                "tool_calls": [
                    {
                        "tool": "replace_function",
                        "args": {
                            "function_name": "foo",
                            "new_code": new_code,
                        },
                    }
                ],
            }
            response = json.dumps(json_payload)

            config = MagicMock()
            config.runtime.allow_outside_workspace = False
            state = {}

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                error = _apply_coding_result(response, state, config)
                assert error is None
                assert "return 3" in file_path.read_text()
            finally:
                os.chdir(old_cwd)

    def test_apply_json_multiple_tool_calls(self):
        """Test multiple tool calls in a single JSON payload."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text(
                "def foo():\n    return 1\n\n"
                "def bar():\n    return 2\n"
            )

            json_payload = {
                "thought": "Replace foo and bar.",
                "tool_calls": [
                    {
                        "tool": "replace_function",
                        "args": {
                            "path": "main.py",
                            "function_name": "foo",
                            "new_code": "def foo():\n    return 10",
                        },
                    },
                    {
                        "tool": "replace_function",
                        "args": {
                            "path": "main.py",
                            "function_name": "bar",
                            "new_code": "def bar():\n    return 20",
                        },
                    },
                ],
            }
            response = json.dumps(json_payload)

            config = MagicMock()
            config.runtime.allow_outside_workspace = False
            state = {}

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                error = _apply_coding_result(response, state, config)
                assert error is None
                content = file_path.read_text()
                assert "return 10" in content
                assert "return 20" in content
            finally:
                os.chdir(old_cwd)

    def test_apply_json_empty_tool_calls_returns_error(self):
        """Test that empty tool_calls returns a helpful error."""
        json_payload = {"thought": "Nothing to do.", "tool_calls": []}
        response = json.dumps(json_payload)

        config = MagicMock()
        state = {}

        error = _apply_coding_result(response, state, config)
        assert error is not None
        assert "No 'tool_calls' found" in error
