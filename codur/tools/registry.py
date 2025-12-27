"""
Tool registry utilities for Codur.
"""

from __future__ import annotations

import inspect
import shutil
from typing import Any

import codur.tools as tool_module


def _iter_tool_functions() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    for name in getattr(tool_module, "__all__", []):
        obj = getattr(tool_module, name, None)
        if callable(obj):
            tools[name] = obj
    return tools


def _summary(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _format_signature(func: Any) -> str:
    signature = inspect.signature(func)
    filtered = []
    internal_params = {"root", "allow_outside_root", "state", "config"}
    for name, param in signature.parameters.items():
        if name in internal_params:
            continue
        filtered.append(param.replace(kind=param.kind))
    return str(signature.replace(parameters=filtered))


def _rg_available() -> bool:
    return shutil.which("rg") is not None


def list_tool_directory(state: object | None = None) -> list[dict]:
    """
    List available tools with signatures and short summaries.
    """
    items = []
    has_rg = _rg_available()
    for name, func in sorted(_iter_tool_functions().items()):
        if not has_rg and name == "ripgrep_search":
            continue
        items.append({
            "name": name,
            "signature": _format_signature(func),
            "summary": _summary(getattr(func, "__doc__", None)),
            "module": getattr(func, "__module__", ""),
        })
    return items


def get_tool_help(name: str, state: object | None = None) -> dict:
    """
    Return detailed help for a named tool.
    """
    tools = _iter_tool_functions()
    func = tools.get(name)
    if not func:
        return {"error": f"Unknown tool: {name}"}
    return {
        "name": name,
        "signature": _format_signature(func),
        "doc": inspect.getdoc(func) or "",
        "module": getattr(func, "__module__", ""),
    }


def get_tool_by_name(name: str) -> callable | None:
    """Get tool function by name for schema generation.

    Args:
        name: Tool name (e.g., "read_file")

    Returns:
        Callable function or None if not found
    """
    tools = _iter_tool_functions()
    return tools.get(name)
