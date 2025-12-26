"""Tool injection registry for language-specific enhancements.

This module provides a centralized registry of tool injectors that automatically
suggest/inject language-specific tools when files are read.

Example:
    When read_file detects a Python file, python_ast_dependencies is automatically injected.
    When read_file detects a Markdown file, markdown_outline is automatically injected.
"""

from typing import Optional, List, Dict, Any
from .base import ToolInjector
from .python import PythonToolInjector
from .markdown import MarkdownToolInjector

# Global registry of all tool injectors
_INJECTORS: List[ToolInjector] = [
    PythonToolInjector(),
    MarkdownToolInjector(),
]

# Extension -> Injector mapping (computed once for performance)
_EXTENSION_MAP: Dict[str, ToolInjector] = {}
for injector in _INJECTORS:
    for ext in injector.extensions:
        _EXTENSION_MAP[ext.lower()] = injector


def get_injector_for_file(path: str) -> Optional[ToolInjector]:
    """Get the appropriate tool injector for a file path.

    Args:
        path: File path to check

    Returns:
        ToolInjector instance if one handles this file type, None otherwise

    Example:
        >>> injector = get_injector_for_file("main.py")
        >>> injector.name
        'Python'
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    return _EXTENSION_MAP.get(ext)


def get_all_injectors() -> List[ToolInjector]:
    """Get all registered tool injectors.

    Returns:
        List of all ToolInjector instances in the registry
    """
    return list(_INJECTORS)


def inject_followup_tools(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inject language-specific followup tools after read_file calls.

    This function scans tool calls for read_file operations and automatically
    injects additional language-specific tools based on file extensions.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        Enhanced list with language-specific tools injected

    Example:
        >>> tools = [{"tool": "read_file", "args": {"path": "main.py"}}]
        >>> result = inject_followup_tools(tools)
        >>> len(result)
        2
        >>> result[1]["tool"]
        'python_ast_dependencies'
    """
    # Group read_file calls by injector
    injector_paths: Dict[ToolInjector, List[str]] = {}

    for tool in tool_calls:
        if tool.get("tool") == "read_file":
            path = tool.get("args", {}).get("path", "")
            if isinstance(path, str):
                injector = get_injector_for_file(path)
                if injector:
                    if injector not in injector_paths:
                        injector_paths[injector] = []
                    injector_paths[injector].append(path)

    # Get followup tools from each injector
    result = list(tool_calls)
    for injector, paths in injector_paths.items():
        followup_tools = injector.get_followup_tools(paths)
        result.extend(followup_tools)

    return result


__all__ = [
    "ToolInjector",
    "get_injector_for_file",
    "get_all_injectors",
    "inject_followup_tools",
]
