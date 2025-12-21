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

EXCLUDE_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
DEFAULT_MAX_BYTES = 200_000
DEFAULT_MAX_RESULTS = 200


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


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            yield Path(dirpath) / filename


def _is_text_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            sample = handle.read(2048)
        return b"\x00" not in sample
    except OSError:
        return False


def read_file(
    path: str,
    root: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        return data[:max_bytes] + "\n... [truncated]"
    return data


def write_file(
    path: str,
    content: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"Wrote {len(content)} bytes to {target}"


def append_file(
    path: str,
    content: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(content)
    return f"Appended {len(content)} bytes to {target}"


def delete_file(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    target.unlink()
    return f"Deleted {target}"


def copy_file(
    source: str,
    destination: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = _resolve_path(source, root, allow_outside_root=allow_outside_root)
    destination_path = _resolve_path(destination, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return f"Copied {source_path} to {destination_path}"


def move_file(
    source: str,
    destination: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = _resolve_path(source, root, allow_outside_root=allow_outside_root)
    destination_path = _resolve_path(destination, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(destination_path))
    return f"Moved {source_path} to {destination_path}"


def copy_file_to_dir(
    source: str,
    destination_dir: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = _resolve_path(source, root, allow_outside_root=allow_outside_root)
    dest_dir_path = _resolve_path(destination_dir, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        dest_dir_path.mkdir(parents=True, exist_ok=True)
    destination_path = dest_dir_path / source_path.name
    shutil.copy2(source_path, destination_path)
    return f"Copied {source_path} to {destination_path}"


def move_file_to_dir(
    source: str,
    destination_dir: str,
    root: str | Path | None = None,
    create_dirs: bool = True,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    source_path = _resolve_path(source, root, allow_outside_root=allow_outside_root)
    dest_dir_path = _resolve_path(destination_dir, root, allow_outside_root=allow_outside_root)
    if create_dirs:
        dest_dir_path.mkdir(parents=True, exist_ok=True)
    destination_path = dest_dir_path / source_path.name
    shutil.move(str(source_path), str(destination_path))
    return f"Moved {source_path} to {destination_path}"


def list_files(
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    state: AgentState | None = None,
) -> list[str]:
    root_path = _resolve_root(root)
    results: list[str] = []
    for file_path in _iter_files(root_path):
        results.append(str(file_path.relative_to(root_path)))
        if len(results) >= max_results:
            break
    return results


def list_dirs(
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    state: AgentState | None = None,
) -> list[str]:
    root_path = _resolve_root(root)
    results: list[str] = []
    for dirpath, dirnames, _ in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for dirname in dirnames:
            dir_path = Path(dirpath) / dirname
            results.append(str(dir_path.relative_to(root_path)))
            if len(results) >= max_results:
                return results
    return results


def search_files(
    query: str,
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    case_sensitive: bool = False,
    state: AgentState | None = None,
) -> list[str]:
    root_path = _resolve_root(root)
    results: list[str] = []
    needle = query if case_sensitive else query.lower()
    for file_path in _iter_files(root_path):
        name = str(file_path.relative_to(root_path))
        haystack = name if case_sensitive else name.lower()
        if needle in haystack:
            results.append(name)
            if len(results) >= max_results:
                break
    return results


def grep_files(
    pattern: str,
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    case_sensitive: bool = False,
    state: AgentState | None = None,
) -> list[dict]:
    root_path = _resolve_root(root)
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)
    results: list[dict] = []
    for file_path in _iter_files(root_path):
        if not _is_text_file(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if regex.search(line):
                        results.append({
                            "file": str(file_path.relative_to(root_path)),
                            "line": line_no,
                            "text": line.rstrip(),
                        })
                        if len(results) >= max_results:
                            return results
        except OSError:
            continue
    return results


def replace_in_file(
    path: str,
    pattern: str,
    replacement: str,
    root: str | Path | None = None,
    count: int = 0,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    new_content, num_replaced = re.subn(pattern, replacement, content, count=count)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(new_content)
    return {"path": str(target), "replacements": num_replaced}


def line_count(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    target = _resolve_path(path, root, allow_outside_root=allow_outside_root)
    count = 0
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return {"path": str(target), "lines": count}
