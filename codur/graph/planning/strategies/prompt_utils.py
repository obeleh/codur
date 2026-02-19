"""Prompt helpers for Phase 2 planning."""

from __future__ import annotations

import shutil

from codur.config import CodurConfig


def build_base_prompt(config: CodurConfig) -> str:
    """Build the base planning prompt using tool-based guidance."""
    from codur.utils.config_helpers import require_default_agent
    default_agent = require_default_agent(config)

    return f"""You are Codur, an autonomous coding agent orchestrator.
Your goal is to understand the user's request and either handle it directly or delegate to a specialized agent.

**Rules:**
- Do not ask the user for permission; just act
- If a file is mentioned, READ IT first before delegating
- If you need real-time information, use duckduckgo_search or fetch_webpage
- Default agent for general tasks: {default_agent}

**Critical:** Always hand off to the right agent:
- Use "agent:codur-coding" for code changes, bug fixes, implementations
- Use "agent:codur-explanation" for code explanations

To call an agent use: `agent_call(agent_name, challenge)`
"""


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
                return f'- "{user_request}" → Call {tool_desc}, then agent_call("{agent}", ...)'
            return f'- "{user_request}" → Call {tool_desc}'
        return f'- "{user_request}" → Investigate with tools'

    elif action == "delegate":
        agent = decision.get("agent", "default agent")
        return f'- "{user_request}" → agent_call("{agent}", "<instructions>")'

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


def build_agent_selection_prompt(
    config: CodurConfig,
    planning_summary: str,
    user_request: str,
) -> str:
    """Build a prompt that forces the LLM to select the next agent in JSON mode.

    This is used when the planning loop reaches MAX_PLANNING_STEPS and needs
    the LLM to make a final decision about which agent to delegate to.

    Args:
        config: The Codur configuration containing agent settings.
        planning_summary: A summary of what happened during planning (tool calls, findings).
        user_request: The original user request.

    Returns:
        A prompt string that instructs the LLM to respond with JSON containing
        next_agent and next_step_suggestion.
    """
    from codur.utils.config_helpers import require_default_agent

    default_agent = require_default_agent(config)

    # Get available agents from config
    agent_configs = config.get("agents.configs", {})
    available_agents = [
        f"agent:{name}" for name, cfg in agent_configs.items() if cfg.get("enabled", True)
    ]

    # Get routing hints
    routing = config.get("agents.preferences.routing", {})
    routing_hints = "\n".join(
        f"  - {task_type}: {agent}" for task_type, agent in routing.items()
    )

    return f"""You are Codur, an autonomous coding agent orchestrator.

The planning phase has completed its investigation. Based on the gathered information, you must now decide which agent should handle this task.

## User Request
{user_request}

## Planning Summary
{planning_summary}

## Available Agents
{chr(10).join(f"- {agent}" for agent in available_agents)}

## Routing Hints
{routing_hints}

## Default Agent
{default_agent}

## Instructions
Based on the planning investigation and user request, select the most appropriate agent and provide a suggested next step for that agent.

You MUST respond with valid JSON containing exactly these fields:
- "next_agent": The agent to delegate to (e.g., "agent:codur-coding", "agent:claude_code")
- "next_step_suggestion": A brief description of what the agent should do first

Example response:
{{"next_agent": "agent:codur-coding", "next_step_suggestion": "Implement the validation function in utils.py based on the schema found in config.json"}}

Respond ONLY with the JSON object, no additional text."""
