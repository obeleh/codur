"""
Tool registry utilities for Codur.
"""

from __future__ import annotations

import inspect
import shutil
from typing import Any

import codur.tools as tool_module
from codur.constants import TaskType
from codur.tools.tool_annotations import (
    tool_scenarios,
    get_tool_scenarios,
    ToolSideEffect,
    get_tool_side_effects,
)


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


@tool_scenarios(TaskType.EXPLANATION)
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
            "scenarios": get_tool_scenarios(func),
        })
    return items


@tool_scenarios(TaskType.EXPLANATION)
def list_tools_for_tasks(
    task_types: list[TaskType] | TaskType | None = None,
    *,
    exclude_task_types: list[TaskType] | TaskType | None = None,
    exclude_side_effects: list[ToolSideEffect] | ToolSideEffect | None = None,
    include_unannotated: bool = False,
    state: object | None = None,
) -> list[dict]:
    """List tools that declare compatibility with TaskTypes, with optional exclusions.

    Args:
        task_types: Include tools with these TaskTypes. If None, all tools match this criterion.
        exclude_task_types: Exclude tools with these TaskTypes.
        exclude_side_effects: Exclude tools with these side effects.
        include_unannotated: Include tools without TaskType annotations.
        state: Agent state (unused, for tool interface compatibility).

    Returns:
        List of tool metadata dicts.

    Examples:
        # All verification-relevant tools
        tools = list_tools_for_tasks([TaskType.CODE_VALIDATION, TaskType.FILE_OPERATION])

        # All tools except git operations
        tools = list_tools_for_tasks(exclude_task_types=TaskType.FILE_OPERATION)

        # Safe tools for verification (no FILE_MUTATION side effect)
        tools = list_tools_for_tasks(
            task_types=[TaskType.CODE_VALIDATION, TaskType.FILE_OPERATION],
            exclude_side_effects=ToolSideEffect.FILE_MUTATION
        )
    """
    # Normalize task_types
    if task_types is None:
        task_set = None
    elif isinstance(task_types, TaskType):
        task_set = {task_types}
    else:
        task_set = set(task_types)

    # Normalize exclude_task_types
    if exclude_task_types is None:
        exclude_set: set[TaskType] = set()
    elif isinstance(exclude_task_types, TaskType):
        exclude_set = {exclude_task_types}
    else:
        exclude_set = set(exclude_task_types)

    # Normalize exclude_side_effects
    if exclude_side_effects is None:
        exclude_effects: set[ToolSideEffect] = set()
    elif isinstance(exclude_side_effects, ToolSideEffect):
        exclude_effects = {exclude_side_effects}
    else:
        exclude_effects = set(exclude_side_effects)

    items = []
    has_rg = _rg_available()
    for name, func in sorted(_iter_tool_functions().items()):
        if not has_rg and name == "ripgrep_search":
            continue

        # Get tool metadata
        scenarios = get_tool_scenarios(func)
        side_effects = get_tool_side_effects(func)

        # Check TaskType inclusion
        if task_set is not None:
            if scenarios:
                if not any(task in scenarios for task in task_set):
                    continue
            elif not include_unannotated:
                continue

        # Check TaskType exclusion
        if exclude_set and scenarios:
            if any(task in scenarios for task in exclude_set):
                continue

        # Check side effect exclusion
        if exclude_effects:
            if any(effect in side_effects for effect in exclude_effects):
                continue

        items.append({
            "name": name,
            "signature": _format_signature(func),
            "summary": _summary(getattr(func, "__doc__", None)),
            "module": getattr(func, "__module__", ""),
            "scenarios": scenarios,
            "side_effects": side_effects,
        })
    return items


@tool_scenarios(TaskType.EXPLANATION)
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
