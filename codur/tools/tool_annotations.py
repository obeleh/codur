"""Tool annotations for scenario-aware tool selection."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from codur.constants import TaskType

SCENARIOS_ATTR = "_scenarios"


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
