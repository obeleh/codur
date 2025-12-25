"""Tests for TUI gitignore support."""

import os
import tempfile
from pathlib import Path
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern


def test_gitignore_pattern_matching():
    """Test that gitignore patterns are correctly matched."""
    # Create test patterns
    gitignore_content = """
# Comments
*.pyc
__pycache__/
.venv/
node_modules/
*.egg-info/
.DS_Store
src/temp/
"""

    patterns = [p for p in gitignore_content.splitlines() if p.strip() and not p.strip().startswith('#')]
    spec = PathSpec.from_lines(GitWildMatchPattern, patterns)

    # Test files that should be ignored
    ignored_files = [
        '__pycache__/module.py',
        'src/__pycache__/test.pyc',
        '.venv/lib/python3.10/site-packages/test.py',
        'package.egg-info/PKG-INFO',
        '.DS_Store',
        'test.pyc',
        'src/temp/file.py',
    ]

    for file in ignored_files:
        assert spec.match_file(file), f"{file} should be ignored"

    # Test files that should NOT be ignored
    included_files = [
        'src/main.py',
        'normal_file.py',
        'tests/test_main.py',
        'README.md',
    ]

    for file in included_files:
        assert not spec.match_file(file), f"{file} should not be ignored"


def test_gitignore_loading():
    """Test that .gitignore file can be loaded and parsed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gitignore_path = os.path.join(tmpdir, '.gitignore')
        gitignore_content = """
*.pyc
__pycache__/
.venv/
"""

        # Write .gitignore
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)

        # Load gitignore patterns using the logic from CodurTUI
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = f.read().splitlines()
            patterns = [p for p in patterns if p.strip() and not p.strip().startswith('#')]
            spec = PathSpec.from_lines(GitWildMatchPattern, patterns)

            # Verify it's not None
            assert spec is not None

            # Test a pattern
            assert spec.match_file('test.pyc')
            assert spec.match_file('__pycache__/module.py')
            assert not spec.match_file('main.py')
        except (IOError, OSError):
            assert False, "Failed to load gitignore"


def test_directory_with_trailing_slash():
    """Test that directory patterns work with trailing slashes in the pattern."""
    patterns = ['__pycache__/', '.venv/', 'node_modules/']
    spec = PathSpec.from_lines(GitWildMatchPattern, patterns)

    # In gitignore, patterns with trailing slashes match directories and their contents
    # but the pattern itself needs to be checked with trailing slash
    assert spec.match_file('__pycache__/')
    assert spec.match_file('src/__pycache__/test.py')
    assert spec.match_file('.venv/lib/python3.10')
    assert spec.match_file('node_modules/package/file.js')

    # Without trailing slash, they won't match
    assert not spec.match_file('__pycache__')
    assert not spec.match_file('.venv')
    assert not spec.match_file('node_modules')


if __name__ == '__main__':
    test_gitignore_pattern_matching()
    test_gitignore_loading()
    test_directory_with_trailing_slash()
    print("✓ All gitignore tests passed!")
