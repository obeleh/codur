"""
Project discovery tools for identifying executable entry points and project structure.
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Iterable

from codur.graph.state import AgentState
from codur.utils.path_utils import resolve_root
from codur.utils.ignore_utils import (
    get_config_from_state,
    get_exclude_dirs,
    is_gitignored,
    load_gitignore,
    should_include_hidden,
    should_respect_gitignore,
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


def _iter_python_files(root: Path, config: object | None = None) -> Iterable[Path]:
    """Recursively iterate over Python files in a directory."""
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
            if filename.endswith((".py", ".pyi")):
                yield Path(dirpath) / filename


def _has_main_block(file_path: Path) -> bool:
    """Check if a Python file has an if __name__ == '__main__': block."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        # Look for if __name__ == "__main__": or if __name__ == '__main__':
        return bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", content))
    except (OSError, IOError):
        return False


def discover_entry_points(
    root: str | Path | None = None,
    state: AgentState | None = None,
) -> str:
    """Discover all executable entry points in the project.

    Scans the project for Python files with `if __name__ == "__main__":` blocks,
    which are typically the executable entry points.

    Returns a prioritized list:
    1. "main.py" (standard convention)
    2. "app.py" (alternative convention for multi-file challenges)
    3. Any other files with main blocks

    Args:
        root: Project root directory (defaults to current working directory)
        state: Agent state (optional, for future context)

    Returns:
        String with list of entry points, one per line with priority indicator
    """
    root_path = resolve_root(root)
    config = get_config_from_state(state)

    entry_points = []

    # Check for standard entry points first
    main_py = root_path / "main.py"
    app_py = root_path / "app.py"

    if main_py.exists() and _has_main_block(main_py):
        entry_points.append(("main.py", 1, "Standard convention"))

    if app_py.exists() and _has_main_block(app_py):
        entry_points.append(("app.py", 2, "Multi-file variant"))

    # Find any other executable Python files
    seen = {main_py, app_py}
    for py_file in _iter_python_files(root_path, config):
        if py_file in seen:
            continue
        if _has_main_block(py_file):
            relative_path = py_file.relative_to(root_path)
            entry_points.append((str(relative_path), 3, "Custom entry point"))
            seen.add(py_file)

    if not entry_points:
        return "No executable entry points found (no files with 'if __name__ == \"__main__\":' blocks)"

    # Format result
    lines = ["Discovered entry points (in priority order):"]
    for filename, priority, description in entry_points:
        lines.append(f"[{priority}] {filename} - {description}")

    return "\n".join(lines)


def get_primary_entry_point(
    root: str | Path | None = None,
    state: AgentState | None = None,
) -> str:
    """Get the primary (highest priority) entry point for the project.

    Returns the single best entry point to execute:
    - "main.py" if it exists and has a main block
    - "app.py" if main.py doesn't exist but app.py does
    - The first other executable file found, or None

    Args:
        root: Project root directory (defaults to current working directory)
        state: Agent state (optional)

    Returns:
        Filename of the primary entry point, or error message if none found
    """
    root_path = resolve_root(root)
    config = get_config_from_state(state)

    # Check standard entry points in priority order
    main_py = root_path / "main.py"
    app_py = root_path / "app.py"

    if main_py.exists() and _has_main_block(main_py):
        return "main.py"

    if app_py.exists() and _has_main_block(app_py):
        return "app.py"

    # Find first other executable file
    for py_file in _iter_python_files(root_path, config):
        if py_file not in {main_py, app_py} and _has_main_block(py_file):
            return str(py_file.relative_to(root_path))

    return "Error: No executable entry point found"
