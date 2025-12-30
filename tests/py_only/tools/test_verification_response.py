"""Tests for verification response tool."""

from __future__ import annotations

import pytest

from codur.tools.verification_response import build_verification_response, VerificationResult
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
        assert isinstance(result, VerificationResult)
        assert result.passed is True
        assert result.reasoning == "All tests passed successfully"
        assert result.status == "PASS"
        assert str(result) == "Verification response recorded: PASS"

    def test_fail_response(self):
        """Test FAIL verification response."""
        result = build_verification_response(
            passed=False,
            reasoning="Test case_2 failed",
            expected="Output: 5",
            actual="Output: 6",
            suggestions="Check the factorial formula"
        )
        assert isinstance(result, VerificationResult)
        assert result.passed is False
        assert result.reasoning == "Test case_2 failed"
        assert result.expected == "Output: 5"
        assert result.actual == "Output: 6"
        assert result.suggestions == "Check the factorial formula"
        assert result.status == "FAIL"
        assert str(result) == "Verification response recorded: FAIL"

    def test_minimal_pass(self):
        """Test minimal PASS with only required fields."""
        result = build_verification_response(
            passed=True,
            reasoning="Verification succeeded"
        )
        assert result.passed is True
        assert result.reasoning == "Verification succeeded"
        assert result.expected is None
        assert result.actual is None
        assert result.suggestions is None
        assert "PASS" in str(result)

    def test_optional_fields_none(self):
        """Test that optional fields can be None."""
        result = build_verification_response(
            passed=True,
            reasoning="Success",
            expected=None,
            actual=None,
            suggestions=None
        )
        assert result.passed is True
        assert result.expected is None
        assert result.actual is None
        assert result.suggestions is None
        assert str(result) == "Verification response recorded: PASS"

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
        assert result.passed is True
        assert str(result) == "Verification response recorded: PASS"


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_status_property_pass(self):
        """Test status property returns PASS."""
        result = VerificationResult(passed=True, reasoning="Success")
        assert result.status == "PASS"

    def test_status_property_fail(self):
        """Test status property returns FAIL."""
        result = VerificationResult(passed=False, reasoning="Failed")
        assert result.status == "FAIL"

    def test_str_representation_pass(self):
        """Test string representation for PASS."""
        result = VerificationResult(passed=True, reasoning="Success")
        assert str(result) == "Verification response recorded: PASS"

    def test_str_representation_fail(self):
        """Test string representation for FAIL."""
        result = VerificationResult(passed=False, reasoning="Failed")
        assert str(result) == "Verification response recorded: FAIL"

    def test_all_fields_populated(self):
        """Test result with all fields populated."""
        result = VerificationResult(
            passed=False,
            reasoning="Test failed",
            expected="5",
            actual="6",
            suggestions="Fix the formula"
        )
        assert result.passed is False
        assert result.reasoning == "Test failed"
        assert result.expected == "5"
        assert result.actual == "6"
        assert result.suggestions == "Fix the formula"
