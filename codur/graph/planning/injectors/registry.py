"""Tool injection registry for language-specific enhancements.

This module provides a centralized registry of tool injectors that automatically
suggest/inject language-specific tools when files are read.
"""

from __future__ import annotations

import hashlib
import json
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
    """Get the appropriate tool injector for a file path."""
    import os
    ext = os.path.splitext(path)[1].lower()
    return _EXTENSION_MAP.get(ext)


def get_all_injectors() -> List[ToolInjector]:
    """Get all registered tool injectors."""
    return list(_INJECTORS)


def inject_followup_tools(
    tool_calls: List[Dict[str, Any]],
    *,
    preserve_order: bool = False,
) -> List[Dict[str, Any]]:
    """Inject language-specific followup tools after read_file calls."""
    if preserve_order:
        # Collect read_file paths and group by injector while preserving order
        read_file_paths: List[str] = []
        injector_paths: Dict[ToolInjector, List[str]] = {}
        read_placeholders: List[int] = []
        result: List[Dict[str, Any]] = []

        for tool in tool_calls:
            if tool.get("tool") == "read_file":
                path = tool.get("args", {}).get("path", "")
                if not isinstance(path, str):
                    continue
                read_file_paths.append(path)
                injector = get_injector_for_file(path)
                if injector:
                    if injector not in injector_paths:
                        injector_paths[injector] = []
                    injector_paths[injector].append(path)
                read_placeholders.append(len(result))
                result.append({"__read_file_placeholder__": True})
            else:
                result.append(tool)

        # Combine multiple read_file calls into a single multi-file operation, placed
        # at the position of the first read_file to preserve order.
        if read_file_paths:
            if len(read_file_paths) > 1:
                combined_read = {"tool": "read_files", "args": {"paths": read_file_paths}}
            else:
                combined_read = {"tool": "read_file", "args": {"path": read_file_paths[0]}}
            first_index = read_placeholders[0]
            result[first_index] = combined_read
            for index in reversed(read_placeholders[1:]):
                result.pop(index)
    else:
        # Collect read_file paths and group by injector
        read_file_paths = []
        injector_paths = {}
        non_read_tools = []

        for tool in tool_calls:
            if tool.get("tool") == "read_file":
                path = tool.get("args", {}).get("path", "")
                if isinstance(path, str):
                    read_file_paths.append(path)
                    injector = get_injector_for_file(path)
                    if injector:
                        if injector not in injector_paths:
                            injector_paths[injector] = []
                        injector_paths[injector].append(path)
            else:
                non_read_tools.append(tool)

        # Build result with combined read operations
        result = []

        # Add non-read tools first
        result.extend(non_read_tools)

        # Combine multiple read_file calls into a single multi-file operation
        if len(read_file_paths) > 1:
            result.append({
                "tool": "read_files",
                "args": {"paths": read_file_paths}
            })
        elif len(read_file_paths) == 1:
            result.append({
                "tool": "read_file",
                "args": {"path": read_file_paths[0]}
            })

    # Track seen tool calls by hash to avoid duplicates
    seen_hashes: set = set()
    for tool in result:
        tool_hash = hashlib.sha256(json.dumps(tool, sort_keys=True).encode()).hexdigest()
        seen_hashes.add(tool_hash)

    # Add followup tools, filtering duplicates
    for injector, paths in injector_paths.items():
        followup_tools = injector.get_followup_tools(paths)
        for tool in followup_tools:
            tool_hash = hashlib.sha256(json.dumps(tool, sort_keys=True).encode()).hexdigest()
            if tool_hash not in seen_hashes:
                result.append(tool)
                seen_hashes.add(tool_hash)

    return result
