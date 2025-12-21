"""
Python linting helpers for Codur.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterable

from codur.tools.filesystem import EXCLUDE_DIRS
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


def _iter_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
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


def lint_python_files(
    paths: list[str],
    root: str | Path | None = None,
    max_errors: int = 200,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    root_path = _resolve_root(root)
    errors: list[dict] = []
    checked = 0
    for raw_path in paths:
        target = _resolve_path(raw_path, root_path, allow_outside_root=allow_outside_root)
        checked += 1
        errors.extend(_lint_file(target))
        if len(errors) >= max_errors:
            break
    return {"checked": checked, "errors": errors[:max_errors]}


def lint_python_tree(
    root: str | Path | None = None,
    max_errors: int = 200,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    root_path = _resolve_root(root)
    errors: list[dict] = []
    checked = 0
    for path in _iter_python_files(root_path):
        checked += 1
        errors.extend(_lint_file(path))
        if len(errors) >= max_errors:
            break
    return {"checked": checked, "errors": errors[:max_errors]}
