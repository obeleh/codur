"""
Structured data file tools for Codur (JSON, YAML, INI).
"""

from __future__ import annotations

import json
import configparser
from pathlib import Path
from typing import Any, TypedDict

import yaml

from codur.constants import TaskType
from codur.graph.state import AgentState
from codur.graph.state_operations import get_config
from codur.tools.tool_annotations import (
    ToolContext,
    ToolSideEffect,
    tool_contexts,
    tool_scenarios,
    tool_side_effects,
)
from codur.utils.path_utils import resolve_path, resolve_root
from codur.utils.validation import validate_file_access


class StructuredWriteResult(TypedDict):
    """Result payload for structured write operations."""
    format: str
    path: str
    bytes_written: int


def _set_nested_value(data: Any, key_path: str | list[str], value: Any) -> Any:
    """Set a nested value on a mapping using a dotted or list path."""
    if isinstance(key_path, str):
        parts = [part for part in key_path.split(".") if part]
    else:
        parts = list(key_path)
    if not parts:
        return value
    cursor = data
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError("Cannot set nested value on non-dict container")
        cursor = cursor.setdefault(part, {})
    if not isinstance(cursor, dict):
        raise ValueError("Cannot set nested value on non-dict container")
    cursor[parts[-1]] = value
    return data


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.EXPLANATION, TaskType.FILE_OPERATION)
def read_json(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> Any:
    """Read a JSON file and return parsed data."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config(state),
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


@tool_scenarios(TaskType.EXPLANATION, TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.DOCUMENTATION)
def json_decode(
    content: str,
    state: AgentState | None = None,
) -> Any:
    """Parse JSON content into native data."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON decode error: {str(e)}"
        }


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR)
def write_json(
    path: str,
    data: Any,
    root: str | Path | None = None,
    indent: int = 2,
    sort_keys: bool = False,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> StructuredWriteResult:
    """Write data to a JSON file."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=sort_keys)
        handle.write("\n")
    return {"format": "json", "path": str(target), "bytes_written": target.stat().st_size}


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def set_json_value(
    path: str,
    key_path: str | list[str],
    value: Any,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Set a nested JSON value and write the file back."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config(state),
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    updated = _set_nested_value(data, key_path, value)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(updated, handle, indent=2)
        handle.write("\n")
    return {"path": str(target), "updated": True}


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.EXPLANATION, TaskType.FILE_OPERATION)
def read_yaml(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> Any:
    """Read a YAML file and return parsed data."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config(state),
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    with open(target, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@tool_scenarios(TaskType.EXPLANATION, TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.DOCUMENTATION)
def yaml_decode(
    content: str,
    state: AgentState | None = None,
) -> Any:
    """Parse YAML content into native data."""
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        return {
            "error": f"YAML decode error: {str(e)}"
        }


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR)
def write_yaml(
    path: str,
    data: Any,
    root: str | Path | None = None,
    sort_keys: bool = False,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> StructuredWriteResult:
    """Write data to a YAML file."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=sort_keys)
    return {"format": "yaml", "path": str(target), "bytes_written": target.stat().st_size}


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def set_yaml_value(
    path: str,
    key_path: str | list[str],
    value: Any,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Set a nested YAML value and write the file back."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config(state),
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    with open(target, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    updated = _set_nested_value(data or {}, key_path, value)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(updated, handle, sort_keys=False)
    return {"path": str(target), "updated": True}


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.EXPLANATION, TaskType.FILE_OPERATION)
def read_ini(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Read an INI file into a nested dict."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config(state),
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    parser = configparser.ConfigParser()
    parser.read(target, encoding="utf-8")
    data: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        data[section] = dict(parser.items(section))
    return data


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR)
def write_ini(
    path: str,
    data: dict[str, dict[str, Any]],
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> StructuredWriteResult:
    """Write a nested dict to an INI file."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    for section, values in data.items():
        parser[section] = {key: str(value) for key, value in values.items()}
    with open(target, "w", encoding="utf-8") as handle:
        parser.write(handle)
    return {"format": "ini", "path": str(target), "bytes_written": target.stat().st_size}


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def set_ini_value(
    path: str,
    section: str,
    option: str,
    value: Any,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Set an INI option value and write the file back."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config(state),
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    parser = configparser.ConfigParser()
    parser.read(target, encoding="utf-8")
    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, option, str(value))
    with open(target, "w", encoding="utf-8") as handle:
        parser.write(handle)
    return {"path": str(target), "updated": True}
