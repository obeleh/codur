"""Tests for coding node directive parsing and line-based replacement."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from codur.graph.nodes.coding import (
    _parse_replacement_directive,
    _extract_code_block,
    _apply_coding_result,
)
from codur.tools import read_file


class TestParseReplacementDirective:
    """Tests for directive parsing."""

    def test_parse_function_replacement_directive(self):
        """Test parsing 'Replace function' directive."""
        response = """Replace function title_case with:
```python
def title_case(sentence: str) -> str:
    return sentence.upper()
```"""

        directive = _parse_replacement_directive(response)
        assert directive is not None
        assert directive["operation"] == "replace_function"
        assert directive["target_name"] == "title_case"
        assert "def title_case" in directive["code"]

    def test_parse_class_replacement_directive(self):
        """Test parsing 'Replace class' directive."""
        response = """Replace class MyClass with:
```python
class MyClass:
    def __init__(self):
        self.value = 42
```"""

        directive = _parse_replacement_directive(response)
        assert directive is not None
        assert directive["operation"] == "replace_class"
        assert directive["target_name"] == "MyClass"
        assert "class MyClass" in directive["code"]

    def test_parse_method_replacement_directive(self):
        """Test parsing 'Update method' directive."""
        response = """Update method `MyClass.process` with:
```python
def process(self, data):
    return data.upper()
```"""

        directive = _parse_replacement_directive(response)
        assert directive is not None
        assert directive["operation"] == "replace_method"
        assert directive["target_name"] == "process"
        assert directive["class_name"] == "MyClass"

    def test_parse_full_file_response_no_directive(self):
        """Test that full file response returns a replace_full_file directive."""
        response = """```python
def title_case(sentence: str) -> str:
    return sentence.upper()
```"""

        directive = _parse_replacement_directive(response)
        assert directive is not None
        assert directive["operation"] == "replace_full_file"
        assert "def title_case" in directive["code"]

    def test_parse_explicit_full_file_directive(self):
        """Test parsing explicit 'Replace full file' directive."""
        response = """Replace full file with:
```python
def new_content():
    pass
```"""

        directive = _parse_replacement_directive(response)
        assert directive is not None
        assert directive["operation"] == "replace_full_file"
        assert "def new_content" in directive["code"]

    def test_parse_explicit_entire_file_directive(self):
        """Test parsing explicit 'Replace entire file' directive."""
        response = """Replace entire file with:
```python
def entire_new_content():
    pass
```"""

        directive = _parse_replacement_directive(response)
        assert directive is not None
        assert directive["operation"] == "replace_full_file"
        assert "def entire_new_content" in directive["code"]

    def test_extract_code_block_from_full_file(self):
        """Test code block extraction from full file response."""
        response = """```python
def hello():
    return "world"
```"""

        code = _extract_code_block(response)
        assert code is not None
        assert "def hello" in code
        assert "return" in code


class TestApplyCodingResultWithReplacement:
    """Tests for applying coding results with line-based replacement."""

    def test_apply_function_replacement(self):
        """Test applying a function replacement directive."""
        with TemporaryDirectory() as tmpdir:
            # Create a test file with a function
            test_file = Path(tmpdir) / "main.py"
            original_content = '''MINOR_WORDS = {"a", "the"}

def title_case(sentence: str) -> str:
    """Convert to title case."""
    raise NotImplementedError

def _run_tests():
    pass
'''
            test_file.write_text(original_content)

            # Change to the temp directory
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Create response with function replacement directive
                response = '''Replace function title_case with:
```python
def title_case(sentence: str) -> str:
    """Convert to title case."""
    return sentence.capitalize()
```'''

                # Create minimal mock config
                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the result
                error = _apply_coding_result(response, state, config)

                # Should succeed with no error
                assert error is None, f"Unexpected error: {error}"

                # Verify the file was modified
                new_content = test_file.read_text()
                assert "return sentence.capitalize()" in new_content
                assert "raise NotImplementedError" not in new_content
                assert "MINOR_WORDS" in new_content  # Other content preserved
                assert "_run_tests" in new_content

            finally:
                os.chdir(old_cwd)

    def test_apply_invalid_syntax_returns_error(self):
        """Test that invalid syntax returns an error."""
        with TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "main.py"
            test_file.write_text('def foo():\n    pass\n')

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Create response with invalid syntax
                invalid_code = "def foo(\n    this is invalid syntax!"
                response = f'''Replace function foo with:
```python
{invalid_code}
```'''

                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the result
                error = _apply_coding_result(response, state, config)

                # Should return an error
                assert error is not None
                assert "Syntax error" in error or "Invalid Python syntax" in error
                
            finally:
                os.chdir(old_cwd)

    def test_apply_invalid_syntax_includes_code_in_error(self):
        """Test that invalid syntax error includes the attempted code for context."""
        with TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "main.py"
            test_file.write_text('def foo():\n    pass\n')

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Create response with invalid syntax
                invalid_code = "def broken_func():\n    return 'missing quote"
                response = f"```python\n{invalid_code}\n```"

                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the result
                error = _apply_coding_result(response, state, config)

                # Should return an error containing the code
                assert error is not None
                assert invalid_code in error
                assert "Invalid Python syntax" in error

            finally:
                os.chdir(old_cwd)

    def test_apply_replacement_function_not_found(self):
        """Test that replacement fails gracefully when function not found."""
        with TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "main.py"
            test_file.write_text('def existing_func():\n    pass\n')

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Try to replace a non-existent function
                response = '''Replace function nonexistent with:
```python
def nonexistent():
    return "new"
```'''

                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the result
                error = _apply_coding_result(response, state, config)

                # Should return an error about function not found
                assert error is not None
                assert "Could not find" in error or "function nonexistent" in error

            finally:
                os.chdir(old_cwd)

    def test_apply_full_file_replacement(self):
        """Test that full file replacement (implicit) works."""
        with TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "main.py"
            test_file.write_text('def old():\n    pass\n')

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Create response with full file (no directive)
                response = '''```python
def new():
    return "replaced"
```'''

                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the result
                error = _apply_coding_result(response, state, config)

                # Should succeed
                assert error is None

                # Verify the file was completely replaced
                new_content = test_file.read_text()
                assert "def new" in new_content
                assert "def old" not in new_content

            finally:
                os.chdir(old_cwd)

    def test_apply_explicit_full_file_directive(self):
        """Test that explicit 'Replace full file' directive works."""
        with TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "main.py"
            test_file.write_text('def old():\n    pass\n')

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Create response with explicit full file directive
                response = '''Replace full file with:
```python
def explicit_new():
    return "explicitly replaced"
```'''

                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the result
                error = _apply_coding_result(response, state, config)

                # Should succeed
                assert error is None

                # Verify the file was completely replaced
                new_content = test_file.read_text()
                assert "def explicit_new" in new_content
                assert "def old" not in new_content

            finally:
                os.chdir(old_cwd)

    def test_full_file_replacement_persists_to_disk(self):
        """Test that full file replacement actually persists changes to disk."""
        with TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "main.py"
            test_file.write_text('original content')
            
            # Verify initial state
            assert test_file.read_text() == 'original content'

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Create response with full file replacement
                response = '''```python
def new_function():
    print("New content persisted")
```'''

                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the result
                error = _apply_coding_result(response, state, config)
                
                assert error is None

                # Explicitly read from disk again to verify persistence
                with open("main.py", "r") as f:
                    content_on_disk = f.read()
                
                assert 'def new_function():' in content_on_disk
                assert 'print("New content persisted")' in content_on_disk
                assert 'original content' not in content_on_disk

            finally:
                os.chdir(old_cwd)


class TestCodingAgentCapability:
    """Integration test simulating agent responses."""

    def test_agent_can_use_targeted_replacement(self):
        """Test that agent response with targeted replacement works end-to-end."""
        with TemporaryDirectory() as tmpdir:
            # Create a realistic test file (like title-case challenge)
            test_file = Path(tmpdir) / "main.py"
            original_content = '''MINOR_WORDS = {
    "a", "an", "the", "and", "or", "but", "for", "nor",
    "on", "at", "to", "from", "by", "of", "in", "with",
}


def title_case(sentence: str) -> str:
    """Convert a sentence to title case with these rules:

    1. Trim leading/trailing whitespace and collapse internal whitespace.
    2. Capitalize the first and last word.
    3. Lowercase minor words unless they are the first or last word.
    4. Preserve words that are already all-caps (length >= 2).
    5. For hyphenated words, apply the same rules to each subword.
       The subwords inherit whether the parent word is first/last.
    """
    raise NotImplementedError


def _run_tests() -> None:
    cases = [
        ("the lord of the rings", "The Lord of the Rings"),
        ("a tale of two cities", "A Tale of Two Cities"),
    ]
    failures = []
    for index, (raw, expected) in enumerate(cases, start=1):
        actual = title_case(raw)
        if actual != expected:
            failures.append((index, raw, expected, actual))
        else:
            print(f"case_{index}: PASSED")

    if failures:
        raise SystemExit(1)

    print("ALL TESTS PASSED!")
'''
            test_file.write_text(original_content)

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Simulate agent response with targeted replacement
                agent_response = '''Replace function title_case with:
```python
def title_case(sentence: str) -> str:
    """Convert a sentence to title case with these rules:

    1. Trim leading/trailing whitespace and collapse internal whitespace.
    2. Capitalize the first and last word.
    3. Lowercase minor words unless they are the first or last word.
    4. Preserve words that are already all-caps (length >= 2).
    5. For hyphenated words, apply the same rules to each subword.
       The subwords inherit whether the parent word is first/last.
    """
    words = sentence.split()
    if not words:
        return ""

    result = []
    for i, word in enumerate(words):
        if i == 0 or i == len(words) - 1:
            result.append(word.capitalize())
        else:
            if word.lower() in MINOR_WORDS:
                result.append(word.lower())
            else:
                result.append(word.capitalize())

    return " ".join(result)
```'''

                config = MagicMock()
                config.runtime.allow_outside_workspace = False
                state = {}

                # Apply the agent's response
                error = _apply_coding_result(agent_response, state, config)

                # Should succeed
                assert error is None, f"Unexpected error: {error}"

                # Verify the function was replaced
                new_content = test_file.read_text()
                assert "words = sentence.split()" in new_content
                assert "raise NotImplementedError" not in new_content
                assert "MINOR_WORDS" in new_content  # Constants preserved
                assert "_run_tests" in new_content  # Other functions preserved

                # Verify it's still valid Python
                from codur.tools.validation import validate_python_syntax
                is_valid, error_msg = validate_python_syntax(new_content)
                assert is_valid, f"Generated file has syntax error: {error_msg}"

            finally:
                os.chdir(old_cwd)
