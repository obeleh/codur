"""Tests for verification response tool."""

from __future__ import annotations

import pytest

from codur.tools.meta_tools import build_verification_response, VerificationResult
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
        assert isinstance(result, dict)
        assert result["passed"] is True
        assert result["reasoning"] == "All tests passed successfully"

    def test_fail_response(self):
        """Test FAIL verification response."""
        result = build_verification_response(
            passed=False,
            reasoning="Test case_2 failed",
            expected="Output: 5",
            actual="Output: 6",
            suggestions="Check the factorial formula"
        )
        assert isinstance(result, dict)
        assert result["passed"] is False
        assert result["reasoning"] == "Test case_2 failed"
        assert result["expected"] == "Output: 5"
        assert result["actual"] == "Output: 6"
        assert result["suggestions"] == "Check the factorial formula"

    def test_minimal_pass(self):
        """Test minimal PASS with only required fields."""
        result = build_verification_response(
            passed=True,
            reasoning="Verification succeeded"
        )
        assert result["passed"] is True
        assert result["reasoning"] == "Verification succeeded"
        assert result.get("expected") is None
        assert result.get("actual") is None
        assert result.get("suggestions") is None

    def test_optional_fields_none(self):
        """Test that optional fields can be None."""
        result = build_verification_response(
            passed=True,
            reasoning="Success",
            expected=None,
            actual=None,
            suggestions=None
        )
        assert result["passed"] is True
        assert result["expected"] is None
        assert result["actual"] is None
        assert result["suggestions"] is None

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
        assert result["passed"] is True


class TestVerificationResult:
    """Tests for VerificationResult TypedDict."""

    def test_dict_structure_pass(self):
        """Test TypedDict can be created with required fields."""
        result: VerificationResult = {"passed": True, "reasoning": "Success"}
        assert result["passed"] is True
        assert result["reasoning"] == "Success"

    def test_dict_structure_fail(self):
        """Test TypedDict can be created with all fields."""
        result: VerificationResult = {"passed": False, "reasoning": "Failed"}
        assert result["passed"] is False
        assert result["reasoning"] == "Failed"

    def test_all_fields_populated(self):
        """Test result with all fields populated."""
        result: VerificationResult = {
            "passed": False,
            "reasoning": "Test failed",
            "expected": "5",
            "actual": "6",
            "suggestions": "Fix the formula"
        }
        assert result["passed"] is False
        assert result["reasoning"] == "Test failed"
        assert result["expected"] == "5"
        assert result["actual"] == "6"
        assert result["suggestions"] == "Fix the formula"
