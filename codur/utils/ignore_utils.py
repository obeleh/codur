"""Helpers for classifying ignored or sensitive paths."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Optional

try:  # Optional dependency used by TUI and file discovery helpers.
    from pathspec import PathSpec
    from pathspec.patterns import GitWildMatchPattern
except ImportError:  # pragma: no cover - fallback when pathspec is unavailable
    PathSpec = None
    GitWildMatchPattern = None


DEFAULT_SECRET_GLOBS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.kdbx",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    "id_ecdsa",
    ".aws/credentials",
    ".aws/config",
    ".npmrc",
    ".pypirc",
]

DEFAULT_METADATA_DIRS = {
    ".git",
    ".hg",
    ".svn",
}

DEFAULT_DEPENDENCY_DIRS = {
    ".venv",
    "venv",
    "node_modules",
}

DEFAULT_CACHE_DIRS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}

DEFAULT_IGNORE_DIRS = DEFAULT_METADATA_DIRS | DEFAULT_DEPENDENCY_DIRS | DEFAULT_CACHE_DIRS


def get_exclude_dirs(config: object | None) -> set[str]:
    if config and getattr(config, "tools", None) is not None:
        custom = getattr(config.tools, "exclude_dirs", None)
        if custom:
            return set(custom)
    return set(DEFAULT_IGNORE_DIRS)


def should_include_hidden(config: object | None) -> bool:
    if config and getattr(config, "tools", None) is not None:
        return bool(getattr(config.tools, "include_hidden_files", False))
    return False


def should_respect_gitignore(config: object | None) -> bool:
    if config and getattr(config, "tools", None) is not None:
        return bool(getattr(config.tools, "respect_gitignore", True))
    return True


def get_secret_globs(config: object | None) -> list[str]:
    if config and getattr(config, "tools", None) is not None:
        tool_settings = config.tools
        if hasattr(tool_settings, "secret_globs"):
            custom = list(tool_settings.secret_globs or [])
            if custom:
                return custom
    return list(DEFAULT_SECRET_GLOBS)


def should_allow_secret_read(config: object | None) -> bool:
    if config and getattr(config, "tools", None) is not None:
        return bool(getattr(config.tools, "allow_read_secrets", False))
    return False


def is_secret_path(path: Path, root: Path, globs: Iterable[str]) -> bool:
    try:
        rel = path.relative_to(root)
        rel_path = rel.as_posix()
    except ValueError:
        rel_path = path.as_posix()
    name = path.name
    for pattern in globs:
        if "/" in pattern:
            if fnmatch(rel_path, pattern):
                return True
        else:
            if fnmatch(name, pattern):
                return True
    return False


def guard_secret_read(path: Path, root: Path, config: object | None) -> None:
    if should_allow_secret_read(config):
        return
    if is_secret_path(path, root, get_secret_globs(config)):
        raise ValueError(
            "Reading secret files is disabled by default. "
            "Set tools.allow_read_secrets: true to override."
        )


def is_hidden_path(path: Path) -> bool:
    for part in path.parts:
        if part.startswith(".") and part not in (".", ".."):
            return True
    return False


def load_gitignore(root: Path) -> Optional["PathSpec"]:
    if PathSpec is None or GitWildMatchPattern is None:
        return None
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return None
    try:
        patterns = gitignore_path.read_text(encoding="utf-8").splitlines()
    except (OSError, IOError):
        return None
    patterns = [line for line in patterns if line.strip() and not line.strip().startswith("#")]
    if not patterns:
        return None
    return PathSpec.from_lines(GitWildMatchPattern, patterns)


def is_gitignored(path: Path, root: Path, spec: Optional["PathSpec"], *, is_dir: bool = False) -> bool:
    if spec is None:
        return False
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    rel_path = rel.as_posix()
    if is_dir:
        return spec.match_file(rel_path) or spec.match_file(f"{rel_path}/")
    return spec.match_file(rel_path)


def get_config_from_state(state: object | None) -> object | None:
    if state is None:
        return None
    if hasattr(state, "get_config"):
        return state.get_config()
    if isinstance(state, dict):
        return state.get("config")
    return None
