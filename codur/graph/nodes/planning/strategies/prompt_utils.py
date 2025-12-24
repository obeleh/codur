"""Prompt helpers for Phase 2 planning."""

from __future__ import annotations

import json

from codur.config import CodurConfig
from codur.graph.nodes.planning.prompt_builder import PlanningPromptBuilder


def build_base_prompt(config: CodurConfig) -> str:
    return PlanningPromptBuilder(config).build_system_prompt()


def format_focus_prompt(base_prompt: str, focus: str, detected_files: list[str]) -> str:
    if detected_files:
        file_list = ", ".join(detected_files)
        focus = f"{focus}\nDetected file hints: {file_list}"
    return f"{base_prompt}\n\n{focus}"


def select_example_file(detected_files: list[str], default: str = "main.py") -> str:
    for path in detected_files:
        if path.endswith(".py"):
            return path
    if detected_files:
        return detected_files[0]
    return default


def select_example_files(
    detected_files: list[str],
    default: tuple[str, str] = ("app.py", "utils.py"),
) -> tuple[str, str]:
    if len(detected_files) >= 2:
        return detected_files[0], detected_files[1]
    if len(detected_files) == 1:
        return detected_files[0], default[1]
    return default


def build_example_line(user_request: str, decision: dict) -> str:
    return f"- \"{user_request}\" -> {json.dumps(decision, ensure_ascii=True)}"


def format_examples(examples: list[str]) -> str:
    return "\n".join(examples)


def format_tool_suggestions(tools: list[str]) -> str:
    if not tools:
        return ""
    return f"Suggested tools: {', '.join(tools)}"


def normalize_agent_name(value: object, fallback: str) -> str:
    if isinstance(value, str):
        return value or fallback
    if value is None:
        return fallback
    return str(value)
