"""Tests for Python syntax validation utilities."""

import pytest
from codur.tools.validation import validate_python_syntax


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
