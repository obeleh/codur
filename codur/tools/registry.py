"""
Tool registry utilities for Codur.
"""

from __future__ import annotations

import inspect
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


def list_tool_directory(state: object | None = None) -> list[dict]:
    """
    List available tools with signatures and short summaries.
    """
    items = []
    for name, func in sorted(_iter_tool_functions().items()):
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
