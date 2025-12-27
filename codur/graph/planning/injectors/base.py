"""Base protocol for language-specific tool injectors."""

from typing import Protocol, FrozenSet, List, Dict, Any


class ToolInjector(Protocol):
    """Protocol for language-specific tool injection.

    A ToolInjector defines how to enhance file operations for a specific
    language or file type by automatically suggesting/injecting additional
    tools when files are read.

    Example:
        When a Python file is read, the PythonToolInjector automatically
        injects python_ast_dependencies to provide AST analysis.
    """

    @property
    def extensions(self) -> FrozenSet[str]:
        """File extensions this injector handles (e.g., {'.py', '.pyi'}).

        Returns:
            Frozen set of lowercase file extensions including the dot.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this language/file type.

        Returns:
            Name like 'Python', 'Markdown', 'JavaScript', etc.
        """
        ...

    def get_followup_tools(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Get additional tools to inject when reading these file paths.

        This is called automatically when read_file is detected for files
        matching this injector's extensions.

        Args:
            paths: List of file paths being read

        Returns:
            List of tool calls to inject, e.g.:
            [{"tool": "python_ast_dependencies", "args": {"path": "main.py"}}]
        """
        ...

    def get_planning_tools(self) -> List[str]:
        """Get tool names to suggest in planning prompts.

        These tools are suggested to the LLM planner as relevant options
        for working with this file type.

        Returns:
            List of tool names, e.g. ["python_ast_outline", "python_ast_graph"]
        """
        ...

    def get_example_tool_calls(self, example_path: str) -> List[Dict[str, Any]]:
        """Get example tool calls for planning prompt examples.

        Used in planning prompts to show the LLM how to work with this
        file type.

        Args:
            example_path: Path to use in the example

        Returns:
            List of example tool calls demonstrating language-specific tools
        """
        ...
