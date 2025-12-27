"""Standardized validation utilities for configuration and filesystem checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codur.config import CodurConfig


class ValidationError(ValueError):
    """Base class for validation errors."""


class ConfigError(ValidationError):
    """Configuration validation error."""


class FileError(ValidationError):
    """File/path validation error."""


class PermissionError(ValidationError):
    """Tool permission error."""


def require_config(
    value: Any,
    name: str,
    message: str | None = None,
) -> None:
    """Validate that a config value is set."""
    if not value:
        msg = message or f"Configuration '{name}' must be set"
        raise ConfigError(msg)


def require_file_exists(
    path: Path,
    context: str = "",
    message: str | None = None,
) -> None:
    """Validate that a file exists."""
    if not path.exists():
        if message:
            raise FileError(message)
        ctx = f" ({context})" if context else ""
        raise FileError(f"Path does not exist{ctx}: {path}")


def require_directory_exists(
    path: Path,
    context: str = "",
    message: str | None = None,
) -> None:
    """Validate that a directory exists."""
    if not path.exists():
        if message:
            raise FileError(message)
        ctx = f" ({context})" if context else ""
        raise FileError(f"Directory does not exist{ctx}: {path}")
    if not path.is_dir():
        ctx = f" ({context})" if context else ""
        raise FileError(f"Path is not a directory{ctx}: {path}")


def require_tool_permission(
    config: CodurConfig,
    permission_attr: str,
    tool_name: str,
    message: str | None = None,
) -> None:
    """Validate that a tool has required permissions."""
    parts = permission_attr.split(".")
    value: Any = config
    for part in parts:
        value = getattr(value, part, None)
        if value is None:
            break
    if not value:
        msg = message or f"{tool_name} is disabled in configuration (check {permission_attr})"
        raise PermissionError(msg)


def validate_tool_permission(config: CodurConfig, permission_check: str) -> None:
    """Validate that a tool permission flag is enabled."""
    if not hasattr(config.tools, permission_check):
        raise PermissionError(f"Unknown permission: {permission_check}")
    if not getattr(config.tools, permission_check):
        raise PermissionError(f"Tool operation denied by configuration: {permission_check}")


def validate_file_exists(
    path: Path,
    context: str = "",
    allow_symlinks: bool = True,
) -> None:
    """Validate that a file exists, optionally rejecting symlinks."""
    if not path.exists():
        ctx = f" ({context})" if context else ""
        raise FileError(f"File not found{ctx}: {path}")
    if not allow_symlinks and path.is_symlink():
        raise FileError(f"Symlinks not allowed: {path}")


def validate_within_workspace(
    path: Path,
    root: Path,
    message: str | None = None,
) -> None:
    """Validate that a path is within the workspace root."""
    if not (path == root or root in path.parents):
        msg = message or f"Path escapes workspace root: {path}"
        raise FileError(msg)


def validate_file_access(
    path: Path,
    root: Path,
    config: CodurConfig | None,
    *,
    operation: str = "read",
    allow_outside_root: bool = False,
    allow_symlinks: bool = True,
) -> None:
    """Validate file access including existence, workspace bounds, and secret guard."""
    if operation == "read":
        validate_file_exists(path, context=f"for {operation}", allow_symlinks=allow_symlinks)
    if not allow_outside_root:
        validate_within_workspace(path, root)
    if operation == "read":
        from codur.utils.ignore_utils import guard_secret_read
        guard_secret_read(path, root, config)
