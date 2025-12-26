"""Tool injector for Markdown documentation files."""

from typing import FrozenSet, List, Dict, Any


class MarkdownToolInjector:
    """Tool injector for Markdown documentation (.md, .markdown).

    Automatically injects markdown analysis tools when Markdown files are read,
    enabling the agent to understand document structure, extract sections,
    and parse tables without manual tool calls.
    """

    @property
    def extensions(self) -> FrozenSet[str]:
        """Markdown file extensions."""
        return frozenset({".md", ".markdown"})

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "Markdown"

    def get_followup_tools(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Inject markdown outline tool for structure analysis.

        Injects markdown_outline for each markdown file to show document
        structure via headers.

        Args:
            paths: List of Markdown file paths being read

        Returns:
            List with markdown_outline tool call(s)
        """
        return [
            {"tool": "markdown_outline", "args": {"path": path}}
            for path in paths
        ]

    def get_planning_tools(self) -> List[str]:
        """Tools to suggest in planning prompts for Markdown files."""
        return [
            "markdown_outline",
            "markdown_extract_sections",
            "markdown_extract_tables",
        ]

    def get_example_tool_calls(self, example_path: str) -> List[Dict[str, Any]]:
        """Example showing read_file + outline for Markdown.

        Args:
            example_path: Markdown file path to use in example

        Returns:
            List of example tool calls
        """
        return [
            {"tool": "read_file", "args": {"path": example_path}},
            {"tool": "markdown_outline", "args": {"path": example_path}}
        ]
