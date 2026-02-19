"""
Ripgrep-backed search tools for Codur.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional

from ripgrepy import Ripgrepy, RipGrepNotFound

from codur.constants import DEFAULT_MAX_RESULTS, TaskType
from codur.graph.state import AgentState
from codur.graph.state_operations import get_config
from codur.tools.tool_annotations import ToolContext, tool_contexts, tool_scenarios
from codur.utils.ignore_utils import get_exclude_dirs, should_respect_gitignore
from codur.utils.path_utils import resolve_root, resolve_path

_DEFAULT_MAX_DEPTH = 50
_DEFAULT_MAX_COUNT = 10_000


def _resolve_exclude_dirs(state: AgentState | None) -> Iterable[str]:
    """Return exclude directory names from tool state config."""
    config = get_config(state)
    return get_exclude_dirs(config)


def _rg_available() -> bool:
    """Return True if ripgrep is available on PATH."""
    return shutil.which("rg") is not None


def _apply_common_flags(
    rg: Ripgrepy,
    *,
    case_sensitive: bool,
    fixed_strings: bool,
    hidden: bool,
    globs: Iterable[str] | None,
    types: Iterable[str] | None,
    exclude_dirs: Iterable[str],
    respect_gitignore: bool,
) -> None:
    """Apply shared ripgrep flags based on tool parameters."""
    rg.json()
    if not respect_gitignore:
        rg.no_ignore()
    if hidden:
        rg.hidden()
    if fixed_strings:
        rg.fixed_strings()
    if case_sensitive:
        rg.case_sensitive()
    else:
        rg.ignore_case()
    rg.max_depth(_DEFAULT_MAX_DEPTH)
    rg.max_count(_DEFAULT_MAX_COUNT)
    for exclude in exclude_dirs:
        rg.glob(f"!**/{exclude}/**")
    if globs:
        for glob in _expand_globs(globs):
            rg.glob(glob)
    if types:
        for type_name in types:
            rg.type_(type_name)


def _relative_match_path(path_text: str, root_path: Path) -> str:
    """Return path relative to root if possible."""
    if not path_text:
        return path_text
    path = Path(path_text)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(root_path))
    except ValueError:
        return path_text


def _expand_globs(globs: Iterable[str]) -> list[str]:
    """Expand globs with useful prefixed variants."""
    expanded: list[str] = []
    seen: set[str] = set()
    for glob in globs:
        if not glob:
            continue
        for candidate in (glob, *_maybe_prefix_glob(glob)):
            if candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)
    return expanded


def _maybe_prefix_glob(glob: str) -> list[str]:
    """Add **/ prefix for globs that target subpaths."""
    negated = glob.startswith("!")
    raw = glob[1:] if negated else glob
    if raw.startswith(("**/", "/")):
        return []
    if "/" not in raw:
        return []
    prefixed = f"**/{raw.lstrip('./')}"
    return [f"!{prefixed}" if negated else prefixed]


def _parse_ripgrep_json(output: str, root_path: Path, max_results: int) -> list[dict]:
    """Parse ripgrep JSON output into result dicts."""
    if not output.strip():
        return []
    results: list[dict] = []
    errors: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            errors.append(line)
            continue
        if data.get("type") == "error":
            message = data.get("data", {}).get("message") or data.get("data", {}).get("error") or str(data)
            errors.append(message)
            continue
        if data.get("type") != "match":
            continue
        match_data = data.get("data", {})
        results.append({
            "file": _relative_match_path(match_data.get("path", {}).get("text", ""), root_path),
            "line": match_data.get("line_number"),
            "text": (match_data.get("lines", {}).get("text") or "").rstrip(),
        })
        if len(results) >= max_results:
            break
    if errors and not results:
        raise ValueError(f"ripgrep error: {' | '.join(errors)}")
    return results


def _iter_files(root: Path, exclude_dirs: Iterable[str]) -> Iterable[Path]:
    """Yield all files under root excluding named directories."""
    exclude_set = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_set]
        for filename in filenames:
            yield Path(dirpath) / filename


def _is_text_file(path: Path) -> bool:
    """Heuristic check for text files based on null bytes."""
    try:
        with open(path, "rb") as handle:
            sample = handle.read(2048)
        return b"\x00" not in sample
    except OSError:
        return False


def _python_grep_files(
    pattern: str,
    root_path: Path,
    max_results: int,
    case_sensitive: bool,
    exclude_dirs: Iterable[str],
) -> list[dict]:
    """Fallback regex search in Python without ripgrep."""
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)
    results: list[dict] = []
    for file_path in _iter_files(root_path, exclude_dirs):
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


@tool_scenarios(TaskType.EXPLANATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def ripgrep_search(
    pattern: str,
    path: str | Path | None = None,
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    case_sensitive: bool = False,
    fixed_strings: bool = False,
    globs: list[str] | None = None,
    types: list[str] | None = None,
    hidden: bool = False,
    state: AgentState | None = None,
) -> list[dict]:
    """Search files using ripgrep and return match metadata."""
    if max_results <= 0:
        return []
    if path is None:
        root_path = resolve_root(root)
    else:
        root_path = resolve_path(path, root)
    exclude_dirs = _resolve_exclude_dirs(state)
    respect_gitignore = should_respect_gitignore(get_config(state))
    if not _rg_available():
        raise ValueError("ripgrep not found")
    try:
        rg = Ripgrepy(pattern, str(root_path))
    except RipGrepNotFound as exc:
        raise ValueError(str(exc)) from exc
    _apply_common_flags(
        rg,
        case_sensitive=case_sensitive,
        fixed_strings=fixed_strings,
        hidden=hidden,
        globs=globs,
        types=types,
        exclude_dirs=exclude_dirs,
        respect_gitignore=respect_gitignore,
    )
    output = rg.run().as_string
    return _parse_ripgrep_json(output, root_path, max_results)


@tool_contexts(ToolContext.SEARCH)
@tool_scenarios(TaskType.EXPLANATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def grep_files(
    pattern: str,
    path: str | Path | None = None,
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    case_sensitive: bool = False,
    state: AgentState | None = None,
) -> list[dict]:
    """Search file contents for a pattern (ripgrep-backed)."""
    if max_results <= 0:
        return []
    if path is None:
        root_path = resolve_root(root)
    else:
        root_path = resolve_path(path, root)
    exclude_dirs = _resolve_exclude_dirs(state)
    if not _rg_available():
        return _python_grep_files(
            pattern=pattern,
            root_path=root_path,
            max_results=max_results,
            case_sensitive=case_sensitive,
            exclude_dirs=exclude_dirs,
        )
    return ripgrep_search(
        pattern=pattern,
        root=root_path,
        max_results=max_results,
        case_sensitive=case_sensitive,
        fixed_strings=False,
        hidden=True,
        state=state,
    )
