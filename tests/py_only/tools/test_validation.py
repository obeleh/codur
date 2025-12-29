"""Tests for Python syntax validation utilities."""

from __future__ import annotations

from pathlib import Path
import subprocess
from unittest import mock

import pytest

from codur.tools.validation import run_pytest, validate_python_syntax


class TestValidatePythonSyntax:
    """Tests for validate_python_syntax function."""

    def test_valid_simple_function(self):
        """Test valid simple function."""
        code = """
def hello():
    return "world"
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_class(self):
        """Test valid class definition."""
        code = """
class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_complex_code(self):
        """Test valid complex code."""
        code = """
def title_case(sentence: str) -> str:
    words = sentence.split()
    if not words:
        return ""

    result = []
    for i, word in enumerate(words):
        if i == 0 or i == len(words) - 1:
            result.append(word.capitalize())
        else:
            result.append(word.lower())

    return " ".join(result)
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_invalid_syntax_missing_colon(self):
        """Test invalid syntax - missing colon."""
        code = """
def hello()
    return "world"
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None
        assert "Syntax error" in error

    def test_invalid_syntax_indent_error(self):
        """Test invalid syntax - indentation error."""
        code = """
def hello():
return "world"
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None

    def test_invalid_syntax_incomplete_assignment(self):
        """Test invalid syntax - incomplete assignment."""
        code = """
words =
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None

    def test_empty_code(self):
        """Test empty code."""
        code = ""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_whitespace_only(self):
        """Test whitespace-only code."""
        code = "   \n   \n   "
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_import_statement(self):
        """Test valid import statement."""
        code = "from typing import List, Dict"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_invalid_statement_expression(self):
        """Test invalid statement."""
        code = "this is not valid python!"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None


class TestRunPytest:
    """Tests for run_pytest function."""

    def test_missing_pytest_binary(self, tmp_path: Path):
        """Return error when pytest is not available."""
        with mock.patch("subprocess.Popen", side_effect=FileNotFoundError):
            result = run_pytest(root=tmp_path)
        assert result["success"] is False
        assert result["exit_code"] == 127
        assert "pytest not found" in result["error"]

    def test_timeout(self, tmp_path: Path):
        """Return timeout error when pytest runs too long."""
        process = mock.Mock()
        process.communicate.side_effect = subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)
        process.returncode = None
        process.kill = mock.Mock()
        process.wait = mock.Mock()

        with mock.patch("subprocess.Popen", return_value=process):
            result = run_pytest(root=tmp_path, timeout=1)
        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_successful_run(self, tmp_path: Path):
        """Return stdout/stderr and success flag."""
        process = mock.Mock()
        process.communicate.return_value = ("ok\n", "")
        process.returncode = 0

        with mock.patch("subprocess.Popen", return_value=process):
            result = run_pytest(root=tmp_path, paths=["."])
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["stdout"] == "ok"
