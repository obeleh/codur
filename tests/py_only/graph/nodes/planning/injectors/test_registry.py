"""Tests for the tool injector registry system."""

from codur.graph.planning.injectors import (
    get_injector_for_file,
    get_all_injectors,
    inject_followup_tools,
    PythonToolInjector,
    MarkdownToolInjector,
)


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

        # Multiple read_file calls combined into single read_files call + multifile analysis
        assert len(result) == 2
        assert result[0] == {"tool": "read_files", "args": {"paths": ["main.py", "utils.py"]}}
        assert result[1]["tool"] == "python_ast_dependencies_multifile"
        assert result[1]["args"]["paths"] == ["main.py", "utils.py"]

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

        # Multiple read_file calls combined into single read_files + Python AST + Markdown outline
        assert len(result) == 3
        assert result[0] == {"tool": "read_files", "args": {"paths": ["main.py", "README.md"]}}
        # Order of injected tools may vary, so check both exist
        injected_tools = result[1:]
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

        # Multiple read_file calls combined + Python AST (json doesn't get followup)
        assert len(result) == 2
        assert result[0] == {"tool": "read_files", "args": {"paths": ["main.py", "config.json"]}}
        assert result[1]["tool"] == "python_ast_dependencies"

    def test_inject_followup_tools_empty_list(self):
        """Test with empty tool list."""
        result = inject_followup_tools([])
        assert result == []

    def test_inject_followup_tools_with_mixed_tools(self):
        """Test that multiple read_file calls are combined while preserving order."""
        tool_calls = [
            {"tool": "list_files", "args": {}},
            {"tool": "read_file", "args": {"path": "main.py"}},
            {"tool": "grep_files", "args": {"pattern": "test"}},
            {"tool": "read_file", "args": {"path": "utils.py"}}
        ]

        result = inject_followup_tools(tool_calls)

        # Order preserved: list_files, combined read_files at first read position, grep_files, followup
        assert len(result) == 4
        assert result[0] == {"tool": "list_files", "args": {}}
        assert result[1] == {"tool": "read_files", "args": {"paths": ["main.py", "utils.py"]}}
        assert result[2] == {"tool": "grep_files", "args": {"pattern": "test"}}
        # Followup tool for multifile analysis
        assert result[3]["tool"] == "python_ast_dependencies_multifile"

    def test_inject_followup_tools_single_read_stays_read_file(self):
        """Test that single read_file is not converted to read_files."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "main.py"}}
        ]

        result = inject_followup_tools(tool_calls)

        # Single read_file should stay as read_file, not become read_files
        assert result[0]["tool"] == "read_file"
        assert result[0]["args"]["path"] == "main.py"

    def test_inject_followup_tools_multiple_same_file(self):
        """Test that multiple reads of the same file are combined."""
        tool_calls = [
            {"tool": "read_file", "args": {"path": "main.py"}},
            {"tool": "read_file", "args": {"path": "main.py"}}
        ]

        result = inject_followup_tools(tool_calls)

        # Should combine to read_files with same path twice
        assert len(result) == 2
        assert result[0] == {"tool": "read_files", "args": {"paths": ["main.py", "main.py"]}}
        assert result[1]["tool"] == "python_ast_dependencies_multifile"
