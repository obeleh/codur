"""Tests for AST-based utilities."""

import pytest
from codur.tools.ast_utils import (
    find_function_lines,
    find_class_lines,
    find_method_lines,
)


class TestFindFunctionLines:
    """Tests for find_function_lines function."""

    def test_find_simple_function(self):
        """Test finding a simple function."""
        code = """def hello():
    return "world"
"""
        result = find_function_lines(code, "hello")
        assert result is not None
        start, end = result
        assert start == 1
        assert end == 2

    def test_find_function_with_args(self):
        """Test finding a function with arguments."""
        code = """def add(a, b):
    return a + b
"""
        result = find_function_lines(code, "add")
        assert result is not None
        start, end = result
        assert start == 1
        assert end == 2

    def test_find_function_multiline(self):
        """Test finding a multi-line function."""
        code = """def title_case(sentence: str) -> str:
    words = sentence.split()
    if not words:
        return ""

    result = []
    for i, word in enumerate(words):
        result.append(word.capitalize())

    return " ".join(result)
"""
        result = find_function_lines(code, "title_case")
        assert result is not None
        start, end = result
        assert start == 1
        assert end == 10

    def test_find_nonexistent_function(self):
        """Test finding a function that doesn't exist."""
        code = """def hello():
    return "world"
"""
        result = find_function_lines(code, "goodbye")
        assert result is None

    def test_find_function_in_multiple_functions(self):
        """Test finding specific function when multiple exist."""
        code = """def first():
    return 1

def second():
    return 2

def third():
    return 3
"""
        result = find_function_lines(code, "second")
        assert result is not None
        start, end = result
        assert start == 4

    def test_find_function_with_invalid_code(self):
        """Test finding function in invalid Python code."""
        code = """this is not valid python!"""
        result = find_function_lines(code, "hello")
        assert result is None

    def test_find_title_case_from_challenge(self):
        """Test finding title_case function from challenge file structure."""
        # Based on challenges/06-title-case-exceptions/main.py structure
        # Note: Original has intentional syntax errors, this is corrected for testing
        code = """'''Challenge harness for 03-title-case-exceptions.'''

MINOR_WORDS = {
    "a", "an", "the", "and", "or", "but", "for", "nor",
    "on", "at", "to", "from", "by", "of", "in", "with",
}

def title_case(sentence: str) -> str:
    \"\"\"
    Convert a sentence to title case with these rules:

    1. Trim leading/trailing whitespace and collapse internal whitespace.
    2. Split the sentence into words by whitespace.
    3. Preserve words that are already all-caps (length >= 2).
    4. For hyphenated words, split them by hyphen, apply capitalization rules to each subword, and rejoin with hyphens.
       The first subword is considered the "First Word" only if the parent word is the First Word.
       The last subword is considered the "Last Word" only if the parent word is the Last Word.
    5. Capitalize the first and last word of the sentence.
    6. Lowercase minor words unless they are the first or last word.
    7. Capitalize all other words.
    8. Join the processed words back into a sentence with single spaces.
    \"\"\"
    # TODO: implement this function
    raise NotImplementedError("Implement the title_case function according to the specified rules.")

def _run_tests() -> None:
    cases = [
        ("the lord of the rings", "The Lord of the Rings"),
        ("a tale of two cities", "A Tale of Two Cities"),
        ("  the rise of NASA and the GPU  ", "The Rise of NASA and the GPU"),
        ("state-of-the-art design", "State-of-the-Art Design"),
        ("from here to eternity", "From Here to Eternity"),
        ("in the middle of nowhere", "In the Middle of Nowhere"),
    ]

    failures = []
    for index, (raw, expected) in enumerate(cases, start=1):
        actual = title_case(raw)
        if actual != expected:
            failures.append((index, raw, expected, actual))
        else:
            print(f"case_{index}: PASSED")

    if failures:
        for index, raw, expected, actual in failures:
            print(f"case_{index}: FAILED")
            print(f"input={raw!r}")
            print(f"expected={expected!r}")
            print(f"actual={actual!r}")
        raise SystemExit(1)

    print("ALL TESTS PASSED!")

def main() -> None:
    _run_tests()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise
"""
        result = find_function_lines(code, "title_case")
        assert result is not None
        start, end = result
        assert start == 8
        assert end == 24


class TestFindClassLines:
    """Tests for find_class_lines function."""

    def test_find_simple_class(self):
        """Test finding a simple class."""
        code = """class MyClass:
    pass
"""
        result = find_class_lines(code, "MyClass")
        assert result is not None
        start, end = result
        assert start == 1
        assert end == 2

    def test_find_class_with_methods(self):
        """Test finding a class with methods."""
        code = """class MyClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""
        result = find_class_lines(code, "MyClass")
        assert result is not None
        start, end = result
        assert start == 1
        assert end >= 5

    def test_find_nonexistent_class(self):
        """Test finding a class that doesn't exist."""
        code = """class MyClass:
    pass
"""
        result = find_class_lines(code, "OtherClass")
        assert result is None

    def test_find_class_in_multiple_classes(self):
        """Test finding specific class when multiple exist."""
        code = """class First:
    pass

class Second:
    pass

class Third:
    pass
"""
        result = find_class_lines(code, "Second")
        assert result is not None
        start, end = result
        assert start == 4

    def test_find_class_with_invalid_code(self):
        """Test finding class in invalid Python code."""
        code = """this is not valid python!"""
        result = find_class_lines(code, "MyClass")
        assert result is None


class TestFindMethodLines:
    """Tests for find_method_lines function."""

    def test_find_simple_method(self):
        """Test finding a simple method."""
        code = """class MyClass:
    def hello(self):
        return "world"
"""
        result = find_method_lines(code, "MyClass", "hello")
        assert result is not None
        start, end = result
        assert start == 2
        assert end == 3

    def test_find_method_with_args(self):
        """Test finding a method with arguments."""
        code = """class MyClass:
    def add(self, a, b):
        return a + b
"""
        result = find_method_lines(code, "MyClass", "add")
        assert result is not None
        start, end = result
        assert start == 2

    def test_find_init_method(self):
        """Test finding __init__ method."""
        code = """class MyClass:
    def __init__(self, value):
        self.value = value
"""
        result = find_method_lines(code, "MyClass", "__init__")
        assert result is not None
        start, end = result
        assert start == 2

    def test_find_nonexistent_method(self):
        """Test finding a method that doesn't exist."""
        code = """class MyClass:
    def hello(self):
        return "world"
"""
        result = find_method_lines(code, "MyClass", "goodbye")
        assert result is None

    def test_find_method_in_nonexistent_class(self):
        """Test finding a method in a class that doesn't exist."""
        code = """class MyClass:
    def hello(self):
        return "world"
"""
        result = find_method_lines(code, "OtherClass", "hello")
        assert result is None

    def test_find_method_multiple_methods(self):
        """Test finding specific method when multiple exist."""
        code = """class MyClass:
    def first(self):
        return 1

    def second(self):
        return 2

    def third(self):
        return 3
"""
        result = find_method_lines(code, "MyClass", "second")
        assert result is not None
        start, end = result
        assert start == 5

    def test_find_method_with_invalid_code(self):
        """Test finding method in invalid Python code."""
        code = """this is not valid python!"""
        result = find_method_lines(code, "MyClass", "hello")
        assert result is None

    def test_find_method_multiline(self):
        """Test finding a multi-line method."""
        code = """class MyClass:
    def complex_method(self):
        result = []
        for i in range(10):
            result.append(i)
        return result
"""
        result = find_method_lines(code, "MyClass", "complex_method")
        assert result is not None
        start, end = result
        assert start == 2
        assert end >= 6
