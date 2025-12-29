"""
Python linting helpers for Codur.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterable

from codur.constants import TaskType
from codur.utils.ignore_utils import (
    get_config_from_state,
    get_exclude_dirs,
    is_gitignored,
    load_gitignore,
    should_include_hidden,
    should_respect_gitignore,
)
from codur.graph.state import AgentState
from codur.tools.tool_annotations import ToolContext, tool_contexts, tool_scenarios
from codur.utils.path_utils import resolve_root, resolve_path


def _iter_python_files(root: Path, config: object | None = None) -> Iterable[Path]:
    exclude_dirs = get_exclude_dirs(config)
    include_hidden = should_include_hidden(config)
    gitignore_spec = load_gitignore(root) if should_respect_gitignore(config) else None
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        filtered_dirs: list[str] = []
        for dirname in dirnames:
            if dirname in exclude_dirs:
                continue
            if not include_hidden and dirname.startswith("."):
                continue
            rel_path = rel_dir / dirname
            if gitignore_spec and is_gitignored(rel_path, root, gitignore_spec, is_dir=True):
                continue
            filtered_dirs.append(dirname)
        dirnames[:] = filtered_dirs
        for filename in filenames:
            if not include_hidden and filename.startswith("."):
                continue
            rel_path = rel_dir / filename
            if gitignore_spec and is_gitignored(rel_path, root, gitignore_spec, is_dir=False):
                continue
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def _lint_file(path: Path) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [{
            "file": str(path),
            "line": exc.lineno or 0,
            "column": exc.offset or 0,
            "message": exc.msg,
        }]
    except OSError as exc:
        return [{
            "file": str(path),
            "line": 0,
            "column": 0,
            "message": f"Failed to read file: {exc}",
        }]
    return []


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_VALIDATION, TaskType.COMPLEX_REFACTOR)
def lint_python_files(
    paths: list[str],
    root: str | Path | None = None,
    max_errors: int = 200,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    root_path = resolve_root(root)
    errors: list[dict] = []
    checked = 0
    for raw_path in paths:
        target = resolve_path(raw_path, root_path, allow_outside_root=allow_outside_root)
        checked += 1
        errors.extend(_lint_file(target))
        if len(errors) >= max_errors:
            break
    return {"checked": checked, "errors": errors[:max_errors]}


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_VALIDATION, TaskType.COMPLEX_REFACTOR)
def lint_python_tree(
    root: str | Path | None = None,
    max_errors: int = 200,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    errors: list[dict] = []
    checked = 0
    for path in _iter_python_files(root_path, config=config):
        checked += 1
        errors.extend(_lint_file(path))
        if len(errors) >= max_errors:
            break
    return {"checked": checked, "errors": errors[:max_errors]}
