"""Tests for the tool injector registry system."""

import pytest

from codur.graph.nodes.planning.injectors import (
    get_injector_for_file,
    get_all_injectors,
    inject_followup_tools,
)
from codur.graph.nodes.planning.injectors.python import PythonToolInjector
from codur.graph.nodes.planning.injectors.markdown import MarkdownToolInjector


class TestInjectorRegistry:
    """Test the injector registry functions."""

    def test_get_all_injectors(self):
        """Test that all injectors are registered."""
        injectors = get_all_injectors()

        assert len(injectors) >= 2
        assert any(isinstance(inj, PythonToolInjector) for inj in injectors)
        assert any(isinstance(inj, MarkdownToolInjector) for inj in injectors)

    def test_get_injector_for_python_file(self):
        """Test getting injector for .py files."""
        injector = get_injector_for_file("main.py")

        assert injector is not None
        assert isinstance(injector, PythonToolInjector)

    def test_get_injector_for_pyi_file(self):
        """Test getting injector for .pyi files."""
        injector = get_injector_for_file("types.pyi")

        assert injector is not None
        assert isinstance(injector, PythonToolInjector)

    def test_get_injector_for_markdown_file(self):
        """Test getting injector for .md files."""
        injector = get_injector_for_file("README.md")

        assert injector is not None
        assert isinstance(injector, MarkdownToolInjector)

    def test_get_injector_for_markdown_extension(self):
        """Test getting injector for .markdown files."""
        injector = get_injector_for_file("docs.markdown")

        assert injector is not None
        assert isinstance(injector, MarkdownToolInjector)

    def test_get_injector_for_unsupported_file(self):
        """Test that unsupported files return None."""
        injector = get_injector_for_file("config.json")
        assert injector is None

        injector = get_injector_for_file("styles.css")
        assert injector is None

    def test_get_injector_case_insensitive(self):
        """Test that extension matching is case-insensitive."""
        injector = get_injector_for_file("README.MD")
        assert injector is not None
        assert isinstance(injector, MarkdownToolInjector)

        injector = get_injector_for_file("main.PY")
        assert injector is not None
        assert isinstance(injector, PythonToolInjector)

    def test_inject_followup_tools_python_single(self):
        """Test injecting followup tools for a single Python file."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "main.py"}}
        ]

        result = inject_followup_tools(tool_calls)

        assert len(result) == 2
        assert result[0] == {"tool": "read_file", "args": {"path": "main.py"}}
        assert result[1]["tool"] == "python_ast_dependencies"
        assert result[1]["args"] == {"path": "main.py"}

    def test_inject_followup_tools_python_multiple(self):
        """Test injecting followup tools for multiple Python files."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "main.py"}},
            {"tool": "read_file", "args": {"path": "utils.py"}}
        ]

        result = inject_followup_tools(tool_calls)

        assert len(result) == 3
        assert result[0] == {"tool": "read_file", "args": {"path": "main.py"}}
        assert result[1] == {"tool": "read_file", "args": {"path": "utils.py"}}
        assert result[2]["tool"] == "python_ast_dependencies_multifile"
        assert result[2]["args"]["paths"] == ["main.py", "utils.py"]

    def test_inject_followup_tools_markdown(self):
        """Test injecting followup tools for Markdown files."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "README.md"}}
        ]

        result = inject_followup_tools(tool_calls)

        assert len(result) == 2
        assert result[0] == {"tool": "read_file", "args": {"path": "README.md"}}
        assert result[1]["tool"] == "markdown_outline"
        assert result[1]["args"] == {"path": "README.md"}

    def test_inject_followup_tools_mixed_types(self):
        """Test injecting followup tools for mixed file types."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "main.py"}},
            {"tool": "read_file", "args": {"path": "README.md"}}
        ]

        result = inject_followup_tools(tool_calls)

        # Should have original tools + Python AST + Markdown outline
        assert len(result) == 4
        assert result[0] == {"tool": "read_file", "args": {"path": "main.py"}}
        assert result[1] == {"tool": "read_file", "args": {"path": "README.md"}}
        # Order of injected tools may vary, so check both exist
        injected_tools = result[2:]
        tool_names = [t["tool"] for t in injected_tools]
        assert "python_ast_dependencies" in tool_names
        assert "markdown_outline" in tool_names

    def test_inject_followup_tools_no_read_file(self):
        """Test that non-read_file tools are not affected."""
        tool_calls = [
            {"tool": "list_files", "args": {}},
            {"tool": "grep_files", "args": {"pattern": "test"}}
        ]

        result = inject_followup_tools(tool_calls)

        assert len(result) == 2
        assert result == tool_calls

    def test_inject_followup_tools_unsupported_file(self):
        """Test that unsupported files don't get followup tools."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "config.json"}}
        ]

        result = inject_followup_tools(tool_calls)

        assert len(result) == 1
        assert result == tool_calls

    def test_inject_followup_tools_mixed_supported_unsupported(self):
        """Test mixing supported and unsupported files."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "main.py"}},
            {"tool": "read_file", "args": {"path": "config.json"}}
        ]

        result = inject_followup_tools(tool_calls)

        # Original tools + Python AST (json doesn't get followup)
        assert len(result) == 3
        assert result[0] == {"tool": "read_file", "args": {"path": "main.py"}}
        assert result[1] == {"tool": "read_file", "args": {"path": "config.json"}}
        assert result[2]["tool"] == "python_ast_dependencies"

    def test_inject_followup_tools_empty_list(self):
        """Test with empty tool list."""
        result = inject_followup_tools([])
        assert result == []

    def test_inject_followup_tools_malformed_tool(self):
        """Test that malformed tools don't crash the system."""
        tool_calls = [
            {"tool": "read_file", "args": {}},  # Missing path
            {"tool": "read_file"},  # Missing args
            {"args": {"path": "main.py"}}  # Missing tool
        ]

        result = inject_followup_tools(tool_calls)
        # Should return original list without crashing
        assert len(result) == 3
