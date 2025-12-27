"""Utilities for safe configuration access with defaults."""

from __future__ import annotations

from typing import Any, TypeVar

from codur.config import CodurConfig
from codur.constants import DEFAULT_CLI_TIMEOUT, DEFAULT_MAX_ITERATIONS

T = TypeVar("T")


def get_or_default(config: CodurConfig | None, path: str, default: T) -> T:
    """Get nested config value with fallback to a default."""
    if config is None:
        return default
    value: Any = config
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return default
    return value if value is not None else default


def require_config(config: CodurConfig | None, path: str, error_msg: str | None = None) -> Any:
    """Get a required config value or raise a ValueError."""
    value = get_or_default(config, path, None)
    if value is None:
        msg = error_msg or f"Required configuration '{path}' is not set"
        raise ValueError(msg)
    return value


def get_max_iterations(config: CodurConfig | None) -> int:
    """Get max iterations with default from constants."""
    return int(get_or_default(config, "runtime.max_iterations", DEFAULT_MAX_ITERATIONS))


def get_cli_timeout(config: CodurConfig | None) -> int:
    """Get CLI timeout with default from constants."""
    return int(get_or_default(config, "agent_execution.default_cli_timeout", DEFAULT_CLI_TIMEOUT))


def get_default_agent(config: CodurConfig | None) -> str:
    """Get default agent if configured."""
    return get_or_default(config, "agents.preferences.default_agent", "")
