"""
Structured data file tools for Codur (JSON, YAML, INI).
"""

from __future__ import annotations

import json
import configparser
from pathlib import Path
from typing import Any

import yaml

from codur.graph.state import AgentState


def _resolve_root(root: str | Path | None) -> Path:
    return (Path(root) if root else Path.cwd()).resolve()


def _resolve_path(
    path: str,
    root: str | Path | None,
    allow_outside_root: bool = False,
) -> Path:
    root_path = _resolve_root(root)
    raw_path = Path(path)
    target = raw_path if raw_path.is_absolute() else root_path / raw_path
    target = target.resolve()
    if allow_outside_root:
        return target
    if target == root_path or root_path in target.parents:
        return target
    raise ValueError(f"Path escapes workspace root: {path}")


def _set_nested_value(data: Any, key_path: str | list[str], value: Any) -> Any:
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


def read_json(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> Any:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(
    path: str,
    data: Any,
    root: str | Path | None = None,
    indent: int = 2,
    sort_keys: bool = False,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=sort_keys)
        handle.write("\n")
    return f"Wrote JSON to {target}"


def set_json_value(
    path: str,
    key_path: str | list[str],
    value: Any,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    updated = _set_nested_value(data, key_path, value)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(updated, handle, indent=2)
        handle.write("\n")
    return {"path": str(target), "updated": True}


def read_yaml(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> Any:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(
    path: str,
    data: Any,
    root: str | Path | None = None,
    sort_keys: bool = False,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=sort_keys)
    return f"Wrote YAML to {target}"


def set_yaml_value(
    path: str,
    key_path: str | list[str],
    value: Any,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    updated = _set_nested_value(data or {}, key_path, value)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(updated, handle, sort_keys=False)
    return {"path": str(target), "updated": True}


def read_ini(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    parser = configparser.ConfigParser()
    parser.read(target, encoding="utf-8")
    data: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        data[section] = dict(parser.items(section))
    return data


def write_ini(
    path: str,
    data: dict[str, dict[str, Any]],
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    for section, values in data.items():
        parser[section] = {key: str(value) for key, value in values.items()}
    with open(target, "w", encoding="utf-8") as handle:
        parser.write(handle)
    return f"Wrote INI to {target}"


def set_ini_value(
    path: str,
    section: str,
    option: str,
    value: Any,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    parser = configparser.ConfigParser()
    parser.read(target, encoding="utf-8")
    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, option, str(value))
    with open(target, "w", encoding="utf-8") as handle:
        parser.write(handle)
    return {"path": str(target), "updated": True}
