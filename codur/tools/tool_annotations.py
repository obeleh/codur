"""Tool annotations for scenario-aware tool selection."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable

from codur.constants import TaskType

SCENARIOS_ATTR = "_scenarios"
CONTEXT_ATTR = "_tool_contexts"
GUARDS_ATTR = "_tool_guards"
SIDE_EFFECTS_ATTR = "_tool_side_effects"


class ToolContext(Enum):
    """Execution context hints for tool injection."""

    FILESYSTEM = "filesystem"
    SEARCH = "search"
    CONFIG = "config"


class ToolGuard(Enum):
    """Guard hints for tool execution."""

    TEST_OVERWRITE = "test_overwrite"


class ToolSideEffect(Enum):
    """Side effect categories for tools.

    Used to identify tools that perform mutations or external actions,
    enabling safety filters (e.g., verification agents should avoid
    tools with FILE_MUTATION or CODE_EXECUTION side effects).
    """

    FILE_MUTATION = "file_mutation"  # Writes, deletes, or modifies files
    CODE_EXECUTION = "code_execution"  # Runs arbitrary code
    STATE_CHANGE = "state_change"  # Modifies external state (git, env, etc)
    NETWORK = "network"  # Makes network requests


def _normalize_scenarios(values: Iterable[Any]) -> list[TaskType]:
    scenarios: list[TaskType] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            scenarios.extend(_normalize_scenarios(value))
            continue
        if isinstance(value, TaskType):
            scenarios.append(value)
            continue
        raise TypeError(f"tool_scenarios expects TaskType values, got {type(value).__name__}")
    return scenarios


def _normalize_contexts(values: Iterable[Any]) -> list[ToolContext]:
    contexts: list[ToolContext] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            contexts.extend(_normalize_contexts(value))
            continue
        if isinstance(value, ToolContext):
            contexts.append(value)
            continue
        raise TypeError(f"tool_contexts expects ToolContext values, got {type(value).__name__}")
    return contexts


def _normalize_guards(values: Iterable[Any]) -> list[ToolGuard]:
    guards: list[ToolGuard] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            guards.extend(_normalize_guards(value))
            continue
        if isinstance(value, ToolGuard):
            guards.append(value)
            continue
        raise TypeError(f"tool_guards expects ToolGuard values, got {type(value).__name__}")
    return guards


def _normalize_side_effects(values: Iterable[Any]) -> list[ToolSideEffect]:
    side_effects: list[ToolSideEffect] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            side_effects.extend(_normalize_side_effects(value))
            continue
        if isinstance(value, ToolSideEffect):
            side_effects.append(value)
            continue
        raise TypeError(f"tool_side_effects expects ToolSideEffect values, got {type(value).__name__}")
    return side_effects


def tool_scenarios(*scenarios: TaskType) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Annotate a tool with TaskType scenario metadata.

    Usage:
        @tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION)
        def read_file(...):
            ...

    This decorator stores scenarios on the function under `_scenarios`.
    """
    normalized = _normalize_scenarios(scenarios)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        existing = getattr(func, SCENARIOS_ATTR, None)
        if not isinstance(existing, list):
            existing = [] if existing is None else list(existing)
        existing.extend(normalized)
        setattr(func, SCENARIOS_ATTR, existing)
        return func

    return decorator


def tool_contexts(*contexts: ToolContext) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Annotate a tool with execution context metadata."""
    normalized = _normalize_contexts(contexts)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        existing = getattr(func, CONTEXT_ATTR, None)
        if not isinstance(existing, list):
            existing = [] if existing is None else list(existing)
        existing.extend(normalized)
        setattr(func, CONTEXT_ATTR, existing)
        return func

    return decorator


def tool_guards(*guards: ToolGuard) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Annotate a tool with guard metadata."""
    normalized = _normalize_guards(guards)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        existing = getattr(func, GUARDS_ATTR, None)
        if not isinstance(existing, list):
            existing = [] if existing is None else list(existing)
        existing.extend(normalized)
        setattr(func, GUARDS_ATTR, existing)
        return func

    return decorator


def tool_side_effects(*side_effects: ToolSideEffect) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Annotate a tool with side effect metadata.

    Usage:
        @tool_side_effects(ToolSideEffect.FILE_MUTATION, ToolSideEffect.CODE_EXECUTION)
        def replace_function(...):
            ...

    This decorator stores side effects on the function under `_tool_side_effects`.
    Used to identify and filter tools with unsafe side effects.
    """
    normalized = _normalize_side_effects(side_effects)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        existing = getattr(func, SIDE_EFFECTS_ATTR, None)
        if not isinstance(existing, list):
            existing = [] if existing is None else list(existing)
        existing.extend(normalized)
        setattr(func, SIDE_EFFECTS_ATTR, existing)
        return func

    return decorator


def get_tool_scenarios(func: Callable[..., Any]) -> list[TaskType]:
    """Return tool scenario metadata for a function."""
    value = getattr(func, SCENARIOS_ATTR, None)
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return [item for item in value if isinstance(item, TaskType)]
    return [value] if isinstance(value, TaskType) else []


def get_tool_contexts(func: Callable[..., Any]) -> list[ToolContext]:
    """Return tool execution context metadata for a function."""
    value = getattr(func, CONTEXT_ATTR, None)
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return [item for item in value if isinstance(item, ToolContext)]
    return [value] if isinstance(value, ToolContext) else []


def get_tool_guards(func: Callable[..., Any]) -> list[ToolGuard]:
    """Return tool guard metadata for a function."""
    value = getattr(func, GUARDS_ATTR, None)
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, ToolGuard)]
    if isinstance(value, (tuple, set)):
        return [item for item in value if isinstance(item, ToolGuard)]
    return [value] if isinstance(value, ToolGuard) else []


def get_tool_side_effects(func: Callable[..., Any]) -> list[ToolSideEffect]:
    """Return tool side effect metadata for a function."""
    value = getattr(func, SIDE_EFFECTS_ATTR, None)
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, ToolSideEffect)]
    if isinstance(value, (tuple, set)):
        return [item for item in value if isinstance(item, ToolSideEffect)]
    return [value] if isinstance(value, ToolSideEffect) else []
