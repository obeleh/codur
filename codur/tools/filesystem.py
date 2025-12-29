"""
Local filesystem tools for Codur.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Iterable

from codur.graph.state import AgentState
from codur.constants import DEFAULT_MAX_BYTES, DEFAULT_MAX_RESULTS, TaskType
from codur.utils.path_utils import resolve_path, resolve_root
from codur.utils.ignore_utils import (
    get_config_from_state,
    get_exclude_dirs,
    is_gitignored,
    load_gitignore,
    should_include_hidden,
    should_respect_gitignore,
)
from codur.utils.validation import validate_file_access
from codur.tools.tool_annotations import (
    ToolContext,
    ToolGuard,
    ToolSideEffect,
    tool_contexts,
    tool_guards,
    tool_scenarios,
    tool_side_effects,
)

def _filter_dirnames(
    *,
    dirpath: str,
    dirnames: list[str],
    root: Path,
    include_hidden: bool,
    exclude_dirs: set[str],
    gitignore_spec: object | None,
) -> None:
    rel_dir = Path(dirpath).relative_to(root)
    filtered: list[str] = []
    for dirname in dirnames:
        if dirname in exclude_dirs:
            continue
        if not include_hidden and dirname.startswith("."):
            continue
        rel_path = rel_dir / dirname
        if gitignore_spec and is_gitignored(rel_path, root, gitignore_spec, is_dir=True):
            continue
        filtered.append(dirname)
    dirnames[:] = filtered


def _iter_files(root: Path, config: object | None = None) -> Iterable[Path]:
    exclude_dirs = get_exclude_dirs(config)
    include_hidden = should_include_hidden(config)
    gitignore_spec = load_gitignore(root) if should_respect_gitignore(config) else None
    for dirpath, dirnames, filenames in os.walk(root):
        _filter_dirnames(
            dirpath=dirpath,
            dirnames=dirnames,
            root=root,
            include_hidden=include_hidden,
            exclude_dirs=exclude_dirs,
            gitignore_spec=gitignore_spec,
        )
        rel_dir = Path(dirpath).relative_to(root)
        for filename in filenames:
            if not include_hidden and filename.startswith("."):
                continue
            rel_path = rel_dir / filename
            if gitignore_spec and is_gitignored(rel_path, root, gitignore_spec, is_dir=False):
                continue
            yield Path(dirpath) / filename


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(
    TaskType.EXPLANATION,
    TaskType.CODE_FIX,
    TaskType.CODE_GENERATION,
    TaskType.REFACTOR,
    TaskType.FILE_OPERATION,
    TaskType.DOCUMENTATION,
)
def read_file(
    path: str,
    root: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    validate_file_access(
        target,
        root_path,
        config,
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        return data[:max_bytes] + "\n... [truncated]"
    return data


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_guards(ToolGuard.TEST_OVERWRITE)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.DOCUMENTATION)
def write_file(
    path: str,
    content: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"Wrote {len(content)} bytes to {target}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.DOCUMENTATION)
def append_file(
    path: str,
    content: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(content)
    return f"Appended {len(content)} bytes to {target}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION)
def delete_file(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.unlink()
    return f"Deleted {target}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION)
def copy_file(
    source: str,
    destination: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = resolve_path(source, root, allow_outside_root=allow_outside_root)
    destination_path = resolve_path(destination, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return f"Copied {source_path} to {destination_path}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION)
def move_file(
    source: str,
    destination: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = resolve_path(source, root, allow_outside_root=allow_outside_root)
    destination_path = resolve_path(destination, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(destination_path))
    return f"Moved {source_path} to {destination_path}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION)
def copy_file_to_dir(
    source: str,
    destination_dir: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = resolve_path(source, root, allow_outside_root=allow_outside_root)
    dest_dir_path = resolve_path(destination_dir, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        dest_dir_path.mkdir(parents=True, exist_ok=True)
    destination_path = dest_dir_path / source_path.name
    shutil.copy2(source_path, destination_path)
    return f"Copied {source_path} to {destination_path}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION)
def move_file_to_dir(
    source: str,
    destination_dir: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = resolve_path(source, root, allow_outside_root=allow_outside_root)
    dest_dir_path = resolve_path(destination_dir, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        dest_dir_path.mkdir(parents=True, exist_ok=True)
    destination_path = dest_dir_path / source_path.name
    shutil.move(str(source_path), str(destination_path))
    return f"Moved {source_path} to {destination_path}"


@tool_contexts(ToolContext.SEARCH)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.EXPLANATION, TaskType.DOCUMENTATION)
def list_files(
    path: str | Path | None = None,
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    state: AgentState | None = None,
) -> list[str]:
    if path and root:
        raise ValueError("Specify either 'path' or 'root', not both.")
    if path:
        root = path
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    results: list[str] = []
    for file_path in _iter_files(root_path, config):
        results.append(str(file_path.relative_to(root_path)))
        if len(results) >= max_results:
            break
    return results


@tool_contexts(ToolContext.SEARCH)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.EXPLANATION)
def list_dirs(
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    state: AgentState | None = None,
) -> list[str]:
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    exclude_dirs = get_exclude_dirs(config)
    include_hidden = should_include_hidden(config)
    gitignore_spec = load_gitignore(root_path) if should_respect_gitignore(config) else None
    results: list[str] = []
    for dirpath, dirnames, _ in os.walk(root_path):
        _filter_dirnames(
            dirpath=dirpath,
            dirnames=dirnames,
            root=root_path,
            include_hidden=include_hidden,
            exclude_dirs=exclude_dirs,
            gitignore_spec=gitignore_spec,
        )
        for dirname in dirnames:
            dir_path = Path(dirpath) / dirname
            results.append(str(dir_path.relative_to(root_path)))
            if len(results) >= max_results:
                return results
    return results


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.EXPLANATION, TaskType.DOCUMENTATION)
def file_tree(
    path: str | None = None,
    root: str | Path | None = None,
    max_depth: int = 3,
    max_results: int = DEFAULT_MAX_RESULTS,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> list[str]:
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    exclude_dirs = get_exclude_dirs(config)
    include_hidden = should_include_hidden(config)
    gitignore_spec = load_gitignore(root_path) if should_respect_gitignore(config) else None
    if path:
        target = resolve_path(path, root_path, allow_outside_root=allow_outside_root)
    else:
        target = root_path
    if target.is_file():
        return [str(target)]

    results: list[str] = []
    base = target
    for dirpath, dirnames, filenames in os.walk(base):
        rel_dir = Path(dirpath).relative_to(base)
        rel_dir_root = Path(dirpath).relative_to(root_path)
        depth = 0 if rel_dir == Path(".") else len(rel_dir.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        _filter_dirnames(
            dirpath=dirpath,
            dirnames=dirnames,
            root=root_path,
            include_hidden=include_hidden,
            exclude_dirs=exclude_dirs,
            gitignore_spec=gitignore_spec,
        )
        for dirname in dirnames:
            results.append(str((Path(dirpath) / dirname).relative_to(base)) + "/")
            if len(results) >= max_results:
                return results
        for filename in filenames:
            if not include_hidden and filename.startswith("."):
                continue
            rel_path = rel_dir_root / filename
            if gitignore_spec and is_gitignored(rel_path, root_path, gitignore_spec, is_dir=False):
                continue
            results.append(str((Path(dirpath) / filename).relative_to(base)))
            if len(results) >= max_results:
                return results
    return results


@tool_contexts(ToolContext.SEARCH)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.EXPLANATION)
def search_files(
    query: str,
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    case_sensitive: bool = False,
    state: AgentState | None = None,
) -> list[str]:
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    results: list[str] = []
    needle = query if case_sensitive else query.lower()
    for file_path in _iter_files(root_path, config):
        name = str(file_path.relative_to(root_path))
        haystack = name if case_sensitive else name.lower()
        if needle in haystack:
            results.append(name)
            if len(results) >= max_results:
                break
    return results


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR, TaskType.DOCUMENTATION)
def replace_in_file(
    path: str,
    pattern: str,
    replacement: str,
    root: str | Path | None = None,
    count: int = 0,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    config = get_config_from_state(state)
    validate_file_access(
        target,
        resolve_root(root),
        config,
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    new_content, num_replaced = re.subn(pattern, replacement, content, count=count)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(new_content)
    return {"path": str(target), "replacements": num_replaced}


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.EXPLANATION, TaskType.FILE_OPERATION)
def line_count(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    config = get_config_from_state(state)
    validate_file_access(
        target,
        resolve_root(root),
        config,
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    count = 0
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return {"path": str(target), "lines": count}


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR, TaskType.DOCUMENTATION)
def inject_lines(
    path: str,
    line: int,
    content: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Insert content at a 1-based line number."""
    if line < 1:
        raise ValueError("line must be >= 1")
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    config = get_config_from_state(state)
    validate_file_access(
        target,
        resolve_root(root),
        config,
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if line > len(lines) + 1:
        raise ValueError("line exceeds file length + 1")
    insert_at = line - 1
    insert_lines = _split_content_lines(content)
    lines[insert_at:insert_at] = insert_lines
    target.write_text("".join(lines), encoding="utf-8")
    return {"path": str(target), "line": line, "inserted_lines": len(insert_lines)}


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR, TaskType.DOCUMENTATION)
def replace_lines(
    path: str,
    start_line: int,
    end_line: int,
    content: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """Replace inclusive line range [start_line, end_line] with content."""
    if start_line < 1 or end_line < start_line:
        raise ValueError("start_line must be >= 1 and <= end_line")
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    config = get_config_from_state(state)
    validate_file_access(
        target,
        resolve_root(root),
        config,
        operation="read",
        allow_outside_root=allow_outside_root,
    )
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if start_line > len(lines) + 1:
        raise ValueError("start_line exceeds file length + 1")
    if end_line > len(lines) + 1:
        raise ValueError("end_line exceeds file length + 1")
    start_idx = start_line - 1
    end_idx = min(end_line, len(lines))
    new_lines = _split_content_lines(content)
    lines[start_idx:end_idx] = new_lines
    target.write_text("".join(lines), encoding="utf-8")
    return {
        "path": str(target),
        "start_line": start_line,
        "end_line": end_line,
        "replaced_lines": end_idx - start_idx,
    }


def _split_content_lines(content: str) -> list[str]:
    if content == "":
        return []
    return content.splitlines(keepends=True)
