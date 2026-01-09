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


def get_max_iterations(config: CodurConfig | None) -> int:
    """Get max iterations with default from constants."""
    return int(get_or_default(config, "runtime.max_iterations", DEFAULT_MAX_ITERATIONS))


def get_cli_timeout(config: CodurConfig | None) -> int:
    """Get CLI timeout with default from constants."""
    return int(get_or_default(config, "agent_execution.default_cli_timeout", DEFAULT_CLI_TIMEOUT))


def get_default_agent(config: CodurConfig | None) -> str:
    """Get default agent if configured."""
    return get_or_default(config, "agents.preferences.default_agent", "")


def require_default_agent(config: CodurConfig | None) -> str:
    """Get default agent, raising if not configured.

    Args:
        config: Codur configuration

    Returns:
        The configured default agent name

    Raises:
        ValueError: If default_agent is not configured
    """
    agent = get_default_agent(config)
    if not agent:
        raise ValueError("agents.preferences.default_agent must be configured")
    return agent
