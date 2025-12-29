"""Tests for verification response tool."""

from __future__ import annotations

import pytest

from codur.tools.verification_response import build_verification_response
from codur.constants import TaskType
from codur.tools.tool_annotations import get_tool_scenarios


class TestBuildVerificationResponse:
    """Tests for build_verification_response function."""

    def test_pass_response(self):
        """Test PASS verification response."""
        result = build_verification_response(
            passed=True,
            reasoning="All tests passed successfully"
        )
        assert result == "Verification response recorded: PASS"

    def test_fail_response(self):
        """Test FAIL verification response."""
        result = build_verification_response(
            passed=False,
            reasoning="Test case_2 failed",
            expected="Output: 5",
            actual="Output: 6",
            suggestions="Check the factorial formula"
        )
        assert result == "Verification response recorded: FAIL"

    def test_minimal_pass(self):
        """Test minimal PASS with only required fields."""
        result = build_verification_response(
            passed=True,
            reasoning="Verification succeeded"
        )
        assert "PASS" in result

    def test_optional_fields_none(self):
        """Test that optional fields can be None."""
        result = build_verification_response(
            passed=True,
            reasoning="Success",
            expected=None,
            actual=None,
            suggestions=None
        )
        assert result == "Verification response recorded: PASS"

    def test_tool_has_result_verification_annotation(self):
        """Test that tool is annotated with RESULT_VERIFICATION TaskType."""
        scenarios = get_tool_scenarios(build_verification_response)
        assert TaskType.RESULT_VERIFICATION in scenarios

    def test_ignored_parameters(self):
        """Test that root, state, and allow_outside_root are ignored."""
        result = build_verification_response(
            passed=True,
            reasoning="Success",
            root="/some/path",
            state=None,
            allow_outside_root=True
        )
        assert result == "Verification response recorded: PASS"
