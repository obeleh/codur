"""Tests for Markdown tool injector."""

import pytest

from codur.graph.planning.injectors import MarkdownToolInjector


class TestMarkdownToolInjector:
    """Test the Markdown tool injector."""

    @pytest.fixture
    def injector(self):
        """Create a Markdown tool injector."""
        return MarkdownToolInjector()

    def test_extensions(self, injector):
        """Test that Markdown injector handles .md and .markdown files."""
        assert ".md" in injector.extensions
        assert ".markdown" in injector.extensions
        assert len(injector.extensions) == 2

    def test_name(self, injector):
        """Test the injector name."""
        assert injector.name == "Markdown"

    def test_get_followup_tools_single_file(self, injector):
        """Test followup tools for a single Markdown file."""
        paths = ["README.md"]
        tools = injector.get_followup_tools(paths)

        assert len(tools) == 1
        assert tools[0]["tool"] == "markdown_outline"
        assert tools[0]["args"] == {"path": "README.md"}

    def test_get_followup_tools_multiple_files(self, injector):
        """Test followup tools for multiple Markdown files."""
        paths = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md"]
        tools = injector.get_followup_tools(paths)

        assert len(tools) == 3
        assert tools[0]["tool"] == "markdown_outline"
        assert tools[0]["args"] == {"path": "README.md"}
        assert tools[1]["tool"] == "markdown_outline"
        assert tools[1]["args"] == {"path": "CONTRIBUTING.md"}
        assert tools[2]["tool"] == "markdown_outline"
        assert tools[2]["args"] == {"path": "CHANGELOG.md"}

    def test_get_followup_tools_markdown_extension(self, injector):
        """Test followup tools for .markdown extension."""
        paths = ["docs.markdown"]
        tools = injector.get_followup_tools(paths)

        assert len(tools) == 1
        assert tools[0]["tool"] == "markdown_outline"
        assert tools[0]["args"] == {"path": "docs.markdown"}

    def test_get_planning_tools(self, injector):
        """Test the list of planning tools."""
        tools = injector.get_planning_tools()

        assert "markdown_outline" in tools
        assert "markdown_extract_sections" in tools
        assert "markdown_extract_tables" in tools

    def test_get_example_tool_calls(self, injector):
        """Test example tool calls for planning prompts."""
        example_path = "README.md"
        tools = injector.get_example_tool_calls(example_path)

        assert len(tools) == 2
        assert tools[0]["tool"] == "read_file"
        assert tools[0]["args"] == {"path": "README.md"}
        assert tools[1]["tool"] == "markdown_outline"
        assert tools[1]["args"] == {"path": "README.md"}

    def test_get_example_tool_calls_with_path(self, injector):
        """Test example tool calls with a different path."""
        example_path = "docs/guide.md"
        tools = injector.get_example_tool_calls(example_path)

        assert tools[0]["args"]["path"] == "docs/guide.md"
        assert tools[1]["args"]["path"] == "docs/guide.md"
