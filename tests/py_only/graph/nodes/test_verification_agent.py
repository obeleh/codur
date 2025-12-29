"""Tests for verification agent parsing logic."""

from __future__ import annotations

from unittest import mock

import pytest
from langchain_core.messages import AIMessage

from codur.graph.execution.verification_agent import _parse_verification_result


class TestParseVerificationResult:
    """Tests for _parse_verification_result function."""

    def test_parse_from_tool_call_pass(self):
        """Test parsing PASS from build_verification_response tool call."""
        execution_result = mock.Mock()
        execution_result.results = [
            {
                "tool": "build_verification_response",
                "args": {
                    "passed": True,
                    "reasoning": "All tests passed successfully"
                }
            }
        ]

        result = _parse_verification_result([], execution_result)

        assert result["passed"] is True
        assert result["reasoning"] == "All tests passed successfully"
        assert result["expected"] is None
        assert result["actual"] is None
        assert result["suggestions"] is None

    def test_parse_from_tool_call_fail(self):
        """Test parsing FAIL from build_verification_response tool call."""
        execution_result = mock.Mock()
        execution_result.results = [
            {
                "tool": "build_verification_response",
                "args": {
                    "passed": False,
                    "reasoning": "Test case_2 failed",
                    "expected": "Output: 5",
                    "actual": "Output: 6",
                    "suggestions": "Check the factorial formula"
                }
            }
        ]

        result = _parse_verification_result([], execution_result)

        assert result["passed"] is False
        assert result["reasoning"] == "Test case_2 failed"
        assert result["expected"] == "Output: 5"
        assert result["actual"] == "Output: 6"
        assert result["suggestions"] == "Check the factorial formula"

    def test_parse_from_boolean_types(self):
        """Test that boolean values are properly converted."""
        execution_result = mock.Mock()
        execution_result.results = [
            {
                "tool": "build_verification_response",
                "args": {
                    "passed": 1,  # Truthy value
                    "reasoning": "Success"
                }
            }
        ]

        result = _parse_verification_result([], execution_result)
        assert result["passed"] is True

    def test_parse_from_json_in_message(self):
        """Test fallback JSON parsing from AI message content."""
        messages = [
            AIMessage(content="""
Here's my verification result:

```json
{
  "verification": "FAIL",
  "reasoning": "Output mismatch detected",
  "expected": "42",
  "actual": "43"
}
```
""")
        ]

        result = _parse_verification_result(messages, None)

        assert result["passed"] is False
        assert result["reasoning"] == "Output mismatch detected"
        assert result["expected"] == "42"
        assert result["actual"] == "43"

    def test_parse_from_plain_json(self):
        """Test parsing plain JSON without code blocks."""
        messages = [
            AIMessage(content='{"verification": "PASS", "reasoning": "All good"}')
        ]

        result = _parse_verification_result(messages, None)

        assert result["passed"] is True
        assert result["reasoning"] == "All good"

    def test_no_tool_call_no_json(self):
        """Test failure when no tool call and no JSON found."""
        messages = [
            AIMessage(content="Just some random text without structure")
        ]

        result = _parse_verification_result(messages, None)

        assert result["passed"] is False
        assert "did not call build_verification_response" in result["reasoning"]

    def test_empty_messages_no_execution_result(self):
        """Test failure with empty messages and no execution result."""
        result = _parse_verification_result([], None)

        assert result["passed"] is False
        assert result["reasoning"] == "No verification response generated"

    def test_multiple_tool_calls_first_match_wins(self):
        """Test that first build_verification_response tool call is used."""
        execution_result = mock.Mock()
        execution_result.results = [
            {
                "tool": "read_file",
                "args": {"path": "test.py"}
            },
            {
                "tool": "build_verification_response",
                "args": {
                    "passed": True,
                    "reasoning": "First call"
                }
            },
            {
                "tool": "build_verification_response",
                "args": {
                    "passed": False,
                    "reasoning": "Second call"
                }
            }
        ]

        result = _parse_verification_result([], execution_result)

        assert result["passed"] is True
        assert result["reasoning"] == "First call"

    def test_no_reasoning_provided(self):
        """Test handling of missing reasoning field."""
        execution_result = mock.Mock()
        execution_result.results = [
            {
                "tool": "build_verification_response",
                "args": {
                    "passed": True
                    # No reasoning provided
                }
            }
        ]

        result = _parse_verification_result([], execution_result)

        assert result["passed"] is True
        assert result["reasoning"] == "No reasoning provided"

    def test_missing_passed_field(self):
        """Test handling of missing passed field defaults to False."""
        execution_result = mock.Mock()
        execution_result.results = [
            {
                "tool": "build_verification_response",
                "args": {
                    "reasoning": "Missing passed field"
                }
            }
        ]

        result = _parse_verification_result([], execution_result)

        # Should default to False when passed field is missing
        assert result["passed"] is False

    def test_multiple_ai_messages_combined(self):
        """Test that multiple AI messages are combined for JSON parsing."""
        messages = [
            AIMessage(content="Let me analyze this."),
            AIMessage(content='{"verification": "PASS", "reasoning": "Looks good"}')
        ]

        result = _parse_verification_result(messages, None)

        assert result["passed"] is True
        assert result["reasoning"] == "Looks good"

    def test_json_in_code_block_with_extra_text(self):
        """Test JSON extraction from message with extra text."""
        messages = [
            AIMessage(content="""
I've verified the implementation.

```json
{
  "verification": "FAIL",
  "reasoning": "Missing edge case handling",
  "suggestions": "Add null check"
}
```

Let me know if you need more details.
""")
        ]

        result = _parse_verification_result(messages, None)

        assert result["passed"] is False
        assert result["reasoning"] == "Missing edge case handling"
        assert result["suggestions"] == "Add null check"
