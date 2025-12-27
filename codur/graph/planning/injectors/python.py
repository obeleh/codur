"""Tool injector for Python source files."""

from typing import FrozenSet, List, Dict, Any


class PythonToolInjector:
    """Tool injector for Python source files (.py, .pyi).

    Automatically injects AST analysis tools when Python files are read,
    enabling the agent to understand code structure, dependencies, and
    relationships without manual tool calls.
    """

    @property
    def extensions(self) -> FrozenSet[str]:
        """Python file extensions."""
        return frozenset({".py", ".pyi"})

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "Python"

    def get_followup_tools(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Inject AST dependency analysis for Python files.

        For a single file, injects python_ast_dependencies.
        For multiple files, injects python_ast_dependencies_multifile.

        Args:
            paths: List of Python file paths being read

        Returns:
            List with AST dependency tool call(s)
        """
        if len(paths) == 1:
            return [{
                "tool": "python_ast_dependencies",
                "args": {"path": paths[0]}
            }]
        else:
            return [{
                "tool": "python_ast_dependencies_multifile",
                "args": {"paths": paths}
            }]

    def get_planning_tools(self) -> List[str]:
        """Tools to suggest in planning prompts for Python files."""
        return [
            "python_ast_outline",
            "python_ast_graph",
            "python_dependency_graph",
            "python_ast_dependencies",
            "python_ast_dependencies_multifile",
        ]

    def get_example_tool_calls(self, example_path: str) -> List[Dict[str, Any]]:
        """Example showing read_file + AST analysis for Python.

        Args:
            example_path: Python file path to use in example

        Returns:
            List of example tool calls
        """
        return [
            {"tool": "read_file", "args": {"path": example_path}},
            {"tool": "python_ast_dependencies", "args": {"path": example_path}}
        ]
