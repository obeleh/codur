"""
Path resolution helpers for workspace-aware tools.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_ROOT: Path | None = None


def set_default_root(root: str | Path | None) -> None:
    """Set a process-wide default workspace root."""
    global _DEFAULT_ROOT
    if root is None:
        _DEFAULT_ROOT = None
    else:
        _DEFAULT_ROOT = (Path(root)).resolve()


def resolve_root(root: str | Path | None) -> Path:
    if root:
        return Path(root).resolve()
    if _DEFAULT_ROOT is not None:
        return _DEFAULT_ROOT
    return Path.cwd().resolve()


def resolve_path(
    path: str,
    root: str | Path | None,
    allow_outside_root: bool = False,
) -> Path:
    root_path = resolve_root(root)
    raw_path = Path(path)
    target = raw_path if raw_path.is_absolute() else root_path / raw_path
    target = target.resolve()
    if allow_outside_root:
        return target
    if target == root_path or root_path in target.parents:
        return target
    raise ValueError(f"Path escapes workspace root: {path}")
