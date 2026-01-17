"""Prompt helpers for Phase 2 planning."""

from __future__ import annotations

import json
import shutil

from codur.config import CodurConfig
from codur.graph.planning.prompt_builder import PlanningPromptBuilder


def format_tool_usage_guidance() -> str:
    """Return standard tool usage guidance for planning prompts."""
    return """**How to Act:**
- Use available tools directly to investigate or complete the task
- Call `delegate_task(agent_name, instructions)` to hand off to a specialized agent
- Call `task_complete(response)` to respond directly to the user
- You may call investigation tools (read_file, list_files, search) before deciding

**Agent Reference:**
- "agent:codur-coding" for code modification, bug fixes, implementations
- "agent:codur-explanation" for code explanation tasks
- "llm:<profile>" for configured LLM profiles (e.g., llm:groq-70b)"""


def build_base_prompt(config: CodurConfig) -> str:
    """Build the base planning prompt using tool-based guidance."""
    from codur.utils.config_helpers import require_default_agent

    default_agent = require_default_agent(config)
    tool_guidance = format_tool_usage_guidance()

    return f"""You are Codur, an autonomous coding agent orchestrator.
Your goal is to understand the user's request and either handle it directly or delegate to a specialized agent.

{tool_guidance}

**Rules:**
- Do not ask the user for permission; just act
- If a file is mentioned, READ IT first before delegating
- If you need real-time information, use duckduckgo_search or fetch_webpage
- Default agent for general tasks: {default_agent}"""


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
    """Convert a decision dict to a tool-based example description."""
    action = decision.get("action")

    if action == "tool":
        tools = decision.get("tool_calls", [])
        if tools:
            tool_desc = ", ".join(f'{t["tool"]}(...)' for t in tools)
            agent = decision.get("agent")
            if agent:
                return f'- "{user_request}" → Call {tool_desc}, then delegate_task("{agent}", ...)'
            return f'- "{user_request}" → Call {tool_desc}'
        return f'- "{user_request}" → Investigate with tools'

    elif action == "delegate":
        agent = decision.get("agent", "default agent")
        return f'- "{user_request}" → delegate_task("{agent}", "<instructions>")'

    elif action == "respond":
        return f'- "{user_request}" → task_complete("<response>")'

    # Fallback for other actions
    return f'- "{user_request}" → Respond directly'


def format_examples(examples: list[str]) -> str:
    return "\n".join(examples)


def format_tool_suggestions(tools: list[str]) -> str:
    if not tools:
        return ""
    filtered = _filter_unavailable_tools(tools)
    if not filtered:
        return ""
    return f"Suggested tools: {', '.join(filtered)}"


def _filter_unavailable_tools(tools: list[str]) -> list[str]:
    if _rg_available():
        return tools
    return [tool for tool in tools if tool != "ripgrep_search"]


def _rg_available() -> bool:
    return shutil.which("rg") is not None


def normalize_agent_name(value: object, fallback: str) -> str:
    if isinstance(value, str):
        return value or fallback
    if value is None:
        return fallback
    return str(value)
