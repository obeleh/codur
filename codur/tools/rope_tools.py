"""Rope-based refactoring and navigation tools for Codur."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from rope.base.project import Project
from rope.contrib import findit
from rope.refactor.extract import ExtractMethod
from rope.refactor.move import MoveModule
from rope.refactor.rename import Rename

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
from codur.utils.path_utils import resolve_root, resolve_path
from codur.utils.validation import (
    require_directory_exists,
    require_file_exists,
    validate_file_access,
    validate_within_workspace,
)


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.REFACTOR, TaskType.CODE_ANALYSIS)
def rope_find_usages(
    path: str,
    line: int | None = None,
    column: int | None = None,
    offset: int | None = None,
    root: str | Path | None = None,
    max_results: int = 200,
    in_hierarchy: bool = False,
    unsure: bool = False,
    resource_paths: list[str] | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Find usages of the symbol at a location using rope."""
    config = get_config(state)
    project, resource, project_root, target, content = _open_project_resource(
        path, root, allow_outside_root, config=config
    )
    try:
        resolved_offset = _offset_from_position(content, line=line, column=column, offset=offset)
        resources = _resources_from_paths(project, project_root, resource_paths)
        locations = findit.find_occurrences(
            project,
            resource,
            resolved_offset,
            unsure=unsure,
            resources=resources,
            in_hierarchy=in_hierarchy,
        )
        occurrences = _locations_to_dicts(project_root, locations, max_results, config=config)
        query_line, query_column = _line_col_from_offset(content, resolved_offset)
        return {
            "root": str(project_root),
            "path": str(target.relative_to(project_root)),
            "absolute_path": str(target),
            "symbol": _extract_identifier(content, resolved_offset),
            "query": {
                "line": query_line,
                "column": query_column,
                "offset": resolved_offset,
            },
            "occurrences": occurrences,
            "count": len(occurrences),
            "truncated": max_results > 0 and len(occurrences) >= max_results,
            "unsure": unsure,
            "in_hierarchy": in_hierarchy,
        }
    finally:
        project.close()


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.REFACTOR, TaskType.CODE_ANALYSIS)
def rope_find_definition(
    path: str,
    line: int | None = None,
    column: int | None = None,
    offset: int | None = None,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Find the definition location for the symbol at a position."""
    config = get_config(state)
    project, resource, project_root, target, content = _open_project_resource(
        path, root, allow_outside_root, config=config
    )
    try:
        resolved_offset = _offset_from_position(content, line=line, column=column, offset=offset)
        location = findit.find_definition(project, content, resolved_offset, resource=resource)
        if not location:
            query_line, query_column = _line_col_from_offset(content, resolved_offset)
            return {
                "root": str(project_root),
                "path": str(target.relative_to(project_root)),
                "absolute_path": str(target),
                "query": {
                    "line": query_line,
                    "column": query_column,
                    "offset": resolved_offset,
                },
                "definition": None,
                "found": False,
            }
        definition = _location_to_dict(project_root, location, config=config)
        return {
            "root": str(project_root),
            "path": str(target.relative_to(project_root)),
            "absolute_path": str(target),
            "definition": definition,
            "found": True,
        }
    finally:
        project.close()


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.REFACTOR, TaskType.CODE_FIX)
def rope_rename_symbol(
    path: str,
    new_name: str,
    symbol: str | None = None,
    line: int | None = None,
    column: int | None = None,
    offset: int | None = None,
    root: str | Path | None = None,
    in_file: bool | None = None,
    in_hierarchy: bool = False,
    unsure: bool | None = None,
    docs: bool = False,
    resource_paths: list[str] | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Rename a symbol using rope and apply changes to the project."""
    if not new_name:
        raise ValueError("new_name is required")
    config = get_config(state)
    project, resource, project_root, target, content = _open_project_resource(
        path, root, allow_outside_root, config=config
    )
    try:
        resolved_offset = _offset_from_position(
            content,
            line=line,
            column=column,
            offset=offset,
            symbol=symbol,
        )
        resources = _resources_from_paths(project, project_root, resource_paths)
        rename = Rename(project, resource, resolved_offset)
        changes = rename.get_changes(
            new_name,
            in_file=in_file,
            in_hierarchy=in_hierarchy,
            unsure=unsure,
            docs=docs,
            resources=resources,
        )
        project.do(changes)
        changed = [res.path for res in changes.get_changed_resources()]
        return {
            "root": str(project_root),
            "path": str(target.relative_to(project_root)),
            "absolute_path": str(target),
            "new_name": new_name,
            "symbol": symbol or _extract_identifier(content, resolved_offset),
            "changed_files": changed,
            "description": changes.get_description(),
        }
    finally:
        project.close()


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.REFACTOR)
def rope_move_module(
    path: str,
    destination_dir: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Move a module file to a destination directory and update imports."""
    if not destination_dir:
        raise ValueError("destination_dir is required")
    config = get_config(state)
    project, resource, project_root, target, _content = _open_project_resource(
        path, root, allow_outside_root, config=config
    )
    try:
        destination = resolve_path(destination_dir, project_root, allow_outside_root=False)
        destination.mkdir(parents=True, exist_ok=True)
        if not destination.is_dir():
            raise ValueError("destination_dir must be a directory")
        relative_dest = destination.relative_to(project_root)
        dest_resource = project.get_folder(str(relative_dest))
        mover = MoveModule(project, resource)
        changes = mover.get_changes(dest_resource)
        project.do(changes)
        changed = [res.path for res in changes.get_changed_resources()]
        return {
            "root": str(project_root),
            "path": str(target.relative_to(project_root)),
            "absolute_path": str(target),
            "destination_dir": str(relative_dest),
            "changed_files": changed,
            "description": changes.get_description(),
        }
    finally:
        project.close()


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.REFACTOR, TaskType.CODE_FIX)
def rope_extract_method(
    path: str,
    extracted_name: str,
    start_line: int | None = None,
    start_column: int | None = None,
    end_line: int | None = None,
    end_column: int | None = None,
    start_offset: int | None = None,
    end_offset: int | None = None,
    root: str | Path | None = None,
    similar: bool = False,
    global_: bool = False,
    kind: str | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Extract a code region into a new method."""
    if not extracted_name:
        raise ValueError("extracted_name is required")
    config = get_config(state)
    project, resource, project_root, target, content = _open_project_resource(
        path, root, allow_outside_root, config=config
    )
    try:
        start = _offset_from_position(
            content,
            line=start_line,
            column=start_column,
            offset=start_offset,
        )
        if end_offset is None and end_line is None:
            raise ValueError("Provide end_offset or end_line/end_column")
        end = _offset_from_position(
            content,
            line=end_line,
            column=end_column,
            offset=end_offset,
        )
        if end <= start:
            raise ValueError("end must be greater than start")
        extractor = ExtractMethod(project, resource, start, end)
        changes = extractor.get_changes(
            extracted_name,
            similar=similar,
            global_=global_,
            kind=kind,
        )
        project.do(changes)
        changed = [res.path for res in changes.get_changed_resources()]
        return {
            "root": str(project_root),
            "path": str(target.relative_to(project_root)),
            "absolute_path": str(target),
            "extracted_name": extracted_name,
            "selection": {
                "start_offset": start,
                "end_offset": end,
            },
            "changed_files": changed,
            "description": changes.get_description(),
        }
    finally:
        project.close()


def _open_project_resource(
    path: str,
    root: str | Path | None,
    allow_outside_root: bool,
    config: object | None = None,
) -> tuple[Project, object, Path, Path, str]:
    """Open a rope Project and resource for a file path."""
    root_path = resolve_root(root)
    require_directory_exists(root_path, message=f"Project root does not exist: {root_path}")
    target = resolve_path(path, root_path, allow_outside_root=allow_outside_root)
    require_file_exists(target, message=f"File not found: {path}")
    validate_file_access(
        target,
        root_path,
        config,
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    project_root = _select_project_root(root_path, target, allow_outside_root)
    project = Project(str(project_root), ropefolder=None)
    try:
        relative = target.relative_to(project_root)
    except ValueError as exc:
        project.close()
        raise ValueError(f"Path is outside project root: {target}") from exc
    resource = project.get_file(str(relative))
    content = target.read_text(encoding="utf-8", errors="replace")
    return project, resource, project_root, target, content


def _select_project_root(root_path: Path, target: Path, allow_outside_root: bool) -> Path:
    """Pick the rope project root based on target and workspace rules."""
    if root_path == target or root_path in target.parents:
        return root_path
    if allow_outside_root:
        return target.parent
    validate_within_workspace(
        target,
        root_path,
        message=f"Path escapes workspace root: {target}",
    )
    return root_path


def _offset_from_position(
    content: str,
    *,
    line: int | None,
    column: int | None,
    offset: int | None,
    symbol: str | None = None,
) -> int:
    """Resolve a byte offset from line/column, offset, or symbol."""
    if offset is not None:
        if offset < 0 or offset > len(content):
            raise ValueError("offset is out of range")
        return offset
    if symbol:
        return _offset_from_symbol(content, symbol)
    if line is None:
        raise ValueError("Provide either offset or line")
    if line < 1:
        raise ValueError("line must be 1 or greater")
    if column is None:
        column = 0
    if column < 0:
        raise ValueError("column must be 0 or greater")
    lines = content.splitlines(keepends=True)
    if line > len(lines):
        raise ValueError("line is out of range")
    line_text = lines[line - 1]
    line_body = line_text[:-1] if line_text.endswith("\n") else line_text
    if column > len(line_body):
        raise ValueError("column is out of range")
    offset_value = sum(len(lines[idx]) for idx in range(line - 1)) + column
    return offset_value


def _line_col_from_offset(content: str, offset: int) -> tuple[int, int]:
    """Convert a byte offset into 1-based line and 0-based column."""
    if offset < 0 or offset > len(content):
        raise ValueError("offset is out of range")
    line = content.count("\n", 0, offset) + 1
    last_newline = content.rfind("\n", 0, offset)
    column = offset if last_newline == -1 else offset - last_newline - 1
    return line, column


def _extract_identifier(content: str, offset: int) -> Optional[str]:
    """Extract a Python identifier surrounding an offset."""
    if not content:
        return None
    if offset >= len(content):
        offset = len(content) - 1
    if offset < 0:
        return None

    if not _is_identifier_char(content[offset]) and offset > 0:
        if _is_identifier_char(content[offset - 1]):
            offset -= 1

    if not _is_identifier_char(content[offset]):
        return None

    start = offset
    while start > 0 and _is_identifier_char(content[start - 1]):
        start -= 1
    end = offset + 1
    while end < len(content) and _is_identifier_char(content[end]):
        end += 1
    if not (content[start].isalpha() or content[start] == "_"):
        return None
    return content[start:end]


def _is_identifier_char(char: str) -> bool:
    """Return True if a character is valid in a Python identifier."""
    return char.isalnum() or char == "_"


def _offset_from_symbol(content: str, symbol: str) -> int:
    """Find the first occurrence of a symbol in content."""
    if not symbol:
        raise ValueError("symbol must be provided")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol):
        raise ValueError("symbol must be a valid identifier")
    match = re.search(rf"\b{re.escape(symbol)}\b", content)
    if not match:
        raise ValueError(f"Symbol '{symbol}' not found in file")
    return match.start()


def _resources_from_paths(
    project: Project,
    project_root: Path,
    resource_paths: list[str] | None,
) -> Optional[list]:
    """Resolve optional resource paths into rope resources."""
    if not resource_paths:
        return None
    resources = []
    for path in resource_paths:
        target = resolve_path(path, project_root, allow_outside_root=False)
        require_file_exists(target, message=f"Resource path not found: {path}")
        relative = target.relative_to(project_root)
        resources.append(project.get_file(str(relative)))
    return resources


def _locations_to_dicts(
    project_root: Path,
    locations: list,
    max_results: int,
    config: object | None = None,
) -> list[dict]:
    """Convert rope location objects to serializable dicts."""
    if max_results > 0:
        locations = locations[:max_results]
    return [_location_to_dict(project_root, loc, config=config) for loc in locations]


def _location_to_dict(project_root: Path, location, config: object | None = None) -> dict:
    """Serialize a single rope location into a dict."""
    path = Path(project_root) / location.resource.path
    validate_file_access(
        path,
        project_root,
        config,
        operation="read",
        allow_outside_root=False,
    )
    content = path.read_text(encoding="utf-8", errors="replace")
    line, column = _line_col_from_offset(content, location.offset)
    start_offset, end_offset = location.region if location.region else (location.offset, location.offset)
    return {
        "path": location.resource.path,
        "absolute_path": str(path),
        "line": line,
        "column": column,
        "offset": location.offset,
        "end_offset": end_offset,
        "unsure": bool(getattr(location, "unsure", False)),
    }
