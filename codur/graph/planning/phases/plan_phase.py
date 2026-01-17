"""Phase 2: Full LLM planning with tool support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.node_types import PlanNodeResult
from codur.graph.state import AgentState
from codur.graph.state_operations import (
    get_iterations,
    get_messages,
    is_verbose,
)
from codur.utils.llm_helpers import create_and_invoke_with_tool_support
from codur.utils.llm_calls import LLMCallLimitExceeded
from codur.utils.config_helpers import require_default_agent
from codur.graph.planning.types import ClassificationResult
from codur.graph.planning.strategies import get_strategy_for_task
from codur.graph.planning.tools import get_planning_tools

if TYPE_CHECKING:
    from codur.graph.planning.prompt_builder import PlanningPromptBuilder

console = Console()

# Maximum investigation steps before forcing a decision
MAX_PLANNING_STEPS = 3


def _extract_tool_calls_from_results(execution_result) -> list[dict]:
    """Extract tool calls from execution results."""
    return [{"tool": r.get("tool"), "args": r.get("args", {})} for r in execution_result.results]


def llm_plan(
    config: CodurConfig,
    prompt_builder: "PlanningPromptBuilder",  # Kept for compatibility, may be removed later
    decision_handler: None,  # No longer used
    json_parser: None,  # No longer used
    state: AgentState,
    llm: None,  # No longer used - LLM created internally
) -> PlanNodeResult:
    """Phase 2: Full LLM planning using tool-based interaction.

    This phase:
    1. First tries strategy.execute() for heuristic shortcuts
    2. If no shortcut, uses tool-based LLM planning
    3. LLM can investigate (read_file, etc.) then decide (delegate_task, task_complete)
    """
    if "config" not in state:
        raise ValueError("AgentState must include config")

    verbose = is_verbose(state)
    if verbose:
        console.print("[bold blue]Planning (Phase 2: Tool-Based LLM Planning)...[/bold blue]")

    messages = get_messages(state)
    iterations = get_iterations(state)

    # Get classification from Phase 1
    classification: ClassificationResult = state.get("classification")
    if not classification:
        raise ValueError("llm_plan requires 'classification' in AgentState")

    # Check for tool results from previous iterations
    from langchain_core.messages import ToolMessage
    tool_results_present = any(isinstance(msg, ToolMessage) for msg in messages)

    # === HEURISTIC SHORTCUTS (via strategy.execute()) ===
    # Strategies can return early if they can resolve without LLM
    strategy = get_strategy_for_task(classification.task_type)
    shortcut_result = strategy.execute(
        classification=classification,
        tool_results_present=tool_results_present,
        messages=messages,
        iterations=iterations,
        config=config,
        verbose=verbose,
    )
    if shortcut_result is not None:
        if verbose:
            console.print("[dim]Strategy resolved via heuristic shortcut[/dim]")
        return shortcut_result

    # === TOOL-BASED LLM PLANNING ===
    # Build context-aware prompt from strategy
    planning_prompt = strategy.build_planning_prompt(classification, config)

    # Prepare tools
    tools = get_planning_tools(include_investigation=True)
    tool_schemas = [convert_to_openai_function(t) for t in tools]

    # Build initial messages for this planning session
    system_msg = SystemMessage(content=planning_prompt)
    planning_messages = [system_msg] + list(messages)  # Will be extended by tool loop

    # Planning loop - allows investigate → investigate → decide flows
    for step in range(MAX_PLANNING_STEPS):
        if verbose:
            console.print(f"[dim]Planning step {step + 1}/{MAX_PLANNING_STEPS}[/dim]")

        try:
            # Invoke LLM with tool support
            # This mutates planning_messages in place, appending AI + Tool messages
            planning_messages, tool_calls, execution_result = create_and_invoke_with_tool_support(
                config=config,
                new_messages=planning_messages,
                tool_schemas=tool_schemas,
                temperature=config.llm.planning_temperature,
                invoked_by="planning.llm_plan",
                state=state,
            )
        except LLMCallLimitExceeded:
            raise
        except Exception as exc:
            if verbose:
                console.print(f"[red]Planning failed: {exc}[/red]")
            # Fallback to default agent
            default_agent = require_default_agent(config)
            return {
                "next_action": "delegate",
                "selected_agent": default_agent,
                "iterations": iterations + 1,
            }

        # Check for control tool calls
        control_result = _check_for_control_action(
            tool_calls,
            planning_messages,
            iterations,
            config,
        )
        if control_result is not None:
            return control_result

        # If no control action but tools were called (e.g., read_file),
        # continue loop - LLM will see tool results on next iteration
        if not tool_calls:
            # LLM didn't call any tool - treat as direct response
            last_content = _get_last_ai_content(planning_messages)
            if verbose:
                console.print("[yellow]LLM responded without using tools[/yellow]")
            return {
                "next_action": "end",
                "final_response": last_content or "Task completed",
                "messages": planning_messages,
                "iterations": iterations + 1,
            }

    # Max steps reached - force delegation to default agent
    if verbose:
        console.print("[yellow]Max planning steps reached, delegating to default[/yellow]")
    default_agent = require_default_agent(config)
    return {
        "next_action": "delegate",
        "selected_agent": default_agent,
        "messages": planning_messages,
        "iterations": iterations + 1,
    }


def _check_for_control_action(
    tool_calls: list[dict],
    messages: list[BaseMessage],
    iterations: int,
    config: CodurConfig,
) -> PlanNodeResult | None:
    """Check if any tool call is a control action (delegate_task, task_complete).

    Returns PlanNodeResult if a control action was found, None otherwise.
    """
    for call in tool_calls:
        name = call.get("name") or call.get("tool")
        args = call.get("args", {})

        if name == "delegate_task":
            agent_name = args.get("agent_name")
            if not agent_name:
                agent_name = require_default_agent(config)
            return {
                "next_action": "delegate",
                "selected_agent": agent_name,
                "delegate_instructions": args.get("instructions", ""),
                "messages": messages,
                "iterations": iterations + 1,
            }

        elif name == "task_complete":
            return {
                "next_action": "end",
                "final_response": args.get("response", "Task completed"),
                "messages": messages,
                "iterations": iterations + 1,
            }

    return None


def _get_last_ai_content(messages: list[BaseMessage]) -> str | None:
    """Extract content from the last AI message."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, str):
                return content
            # Handle list content (some models return list of content blocks)
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                return " ".join(text_parts)
    return None
