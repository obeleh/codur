"""Phase 2: Full LLM planning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.constants import TaskType
from codur.graph.node_types import PlanNodeResult, PlanningDecision
from codur.graph.state import AgentState
from codur.graph.state_operations import (
    get_iterations,
    get_llm_calls,
    get_messages,
    is_verbose,
    get_first_human_message_content_from_messages,
)
from codur.llm import create_llm_profile
from codur.utils.retry import LLMRetryStrategy
from codur.utils.llm_calls import LLMCallLimitExceeded
from codur.utils.config_helpers import require_default_agent
from codur.graph.planning.types import ClassificationResult
from codur.graph.planning.strategies import get_strategy_for_task
from codur.graph.planning.tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
)
from codur.graph.planning.validators import looks_like_change_request
from codur.utils.json_parser import JSONResponseParser


if TYPE_CHECKING:
    from codur.graph.planning.prompt_builder import PlanningPromptBuilder
    from codur.graph.planning.decision_handler import PlanningDecisionHandler

console = Console()


def _format_candidates(candidates, limit: int = 5) -> str:
    if not candidates:
        return "none"
    trimmed = candidates[:limit]
    return ", ".join(f"{item.task_type.value}:{item.confidence:.0%}" for item in trimmed)


def llm_plan(
    config: CodurConfig,
    prompt_builder: "PlanningPromptBuilder",
    decision_handler: "PlanningDecisionHandler",
    json_parser: "JSONResponseParser",
    state: AgentState,
    llm: BaseChatModel,
) -> PlanNodeResult:
    """Phase 2: Full LLM planning for uncertain cases.

    This is the main planning phase that handles cases where:
    - Textual patterns didn't match (Phase 0)
    - Classification confidence was insufficient (Phase 1)

    Uses context-aware prompt based on Phase 1 classification to reduce token usage.
    """
    if "config" not in state:
        raise ValueError("AgentState must include config")

    if is_verbose(state):
        console.print("[bold blue]Planning (Phase 2: Full LLM Planning)...[/bold blue]")

    # TODO: Is llm_calls begin used?
    def _with_llm_calls(result: PlanNodeResult) -> PlanNodeResult:
        result["llm_calls"] = get_llm_calls(state)
        result["next_step_suggestion"] = None
        return result

    messages = get_messages(state)
    iterations = get_iterations(state)

    tool_results_present = any(
        isinstance(msg, ToolMessage)
        for msg in messages
    )

    classification = state.get("classification")
    if not classification:
        raise ValueError("llm_plan requires 'classification' in AgentState")
    prompt_messages = _build_phase2_messages(
        messages, tool_results_present, classification, config
    )

    # If list_files results are present, select a likely python file and read it.
    if (
        tool_results_present
        and not classification.detected_files
        and not tool_results_include_read_file(messages)
    ):
        candidate = select_file_from_tool_results(messages)
        if candidate:
            coding_tasks = {TaskType.CODE_FIX, TaskType.CODE_GENERATION}
            result = {
                "next_action": "tool",
                "tool_calls": [{"tool": "read_file", "args": {"path": candidate}}],
                "iterations": iterations + 1,
            }
            if classification.task_type in coding_tasks:
                result["selected_agent"] = "agent:codur-coding"
            return _with_llm_calls(result)

    # If no file hint is available for a change request, list files for discovery.
    last_human_msg = get_first_human_message_content_from_messages(messages)
    if (
        not tool_results_present
        and not classification.detected_files
        and last_human_msg
        and looks_like_change_request(last_human_msg)
    ):
        return _with_llm_calls({
            "next_action": "tool",
            "tool_calls": [{"tool": "list_files", "args": {}}],
            "selected_agent": "agent:codur-coding",
            "iterations": iterations + 1,
        })

    retry_strategy = LLMRetryStrategy(
        max_attempts=config.planning.max_retry_attempts,
        initial_delay=config.planning.retry_initial_delay,
        backoff_factor=config.planning.retry_backoff_factor,
    )
    try:
        # Create LLM for planning with lower temperature for more deterministic JSON output
        planning_llm = create_llm_profile(
            config,
            config.llm.default_profile,
            json_mode=True,
            temperature=config.llm.planning_temperature
        )

        active_llm, response, profile_name = retry_strategy.invoke_with_fallbacks(
            config,
            planning_llm,
            prompt_messages,
            state=state,
            invoked_by="planning.llm_plan",
        )
        content = response.content
        if is_verbose(state):
            console.print(f"[dim]LLM content: {content}[/dim]")
    except Exception as exc:
        if isinstance(exc, LLMCallLimitExceeded):
            raise
        if "Failed to validate JSON" in str(exc):
            console.print("  PLANNING ERROR - LLM returned invalid JSON", style="red bold on yellow")

        default_agent = require_default_agent(config)
        if is_verbose(state):
            console.print(f"[red]Planning failed: {str(exc)}[/red]")
            console.print(f"[yellow]Falling back to default agent: {default_agent}[/yellow]")
        return _with_llm_calls({
            "next_action": "delegate",
            "selected_agent": default_agent,
            "iterations": iterations + 1,
        })


    try:
        parser = JSONResponseParser()
        decision = parser.parse(content)
        decision = PlanningDecision(**decision)

        if decision is None:
            raise ValueError("Parsed decision is None")

        return _with_llm_calls(decision_handler.handle_decision(decision, iterations, response))

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        console.print("\n" + "=" * 80, style="red bold")
        console.print("  PLANNING ERROR - Failed to parse LLM decision", style="red bold on yellow")
        console.print("=" * 80, style="red bold")
        console.print(f"  Error Type: {type(exc).__name__}", style="red")
        console.print(f"  Error Message: {str(exc)}", style="red")
        console.print(f"  LLM Response: {content[:200]}...", style="yellow")
        console.print("=" * 80 + "\n", style="red bold")

        default_agent = require_default_agent(config)
        console.print(f"  Falling back to default agent: {default_agent}", style="yellow")

        return _with_llm_calls({
            "next_action": "delegate",
            "selected_agent": default_agent,
            "iterations": iterations + 1,
        })


def _build_phase2_messages(
    messages: list[BaseMessage],
    has_tool_results: bool,
    classification: ClassificationResult,
    config: CodurConfig,
) -> list[BaseMessage]:
    """Build prompt messages for Phase 2 with context-aware planning guidance.

    Uses the classification from Phase 1 to build a focused, task-specific prompt
    that guides the LLM through multi-step reasoning (Chain-of-Thought) for
    file-based coding tasks.
    """
    # Use context-aware prompt based on classification to guide multi-step planning
    strategy = get_strategy_for_task(classification.task_type)
    planning_prompt = strategy.build_planning_prompt(classification, config)
    system_message = SystemMessage(content=planning_prompt)
    prompt_messages = [system_message] + list(messages)

    if has_tool_results:
        system_message.content += (
            "Tool results or verification errors are available above. Review them and respond with valid JSON.\n"
            "1. If tool results (like web search or file read) provide the answer → use action: 'respond' with the answer.\n"
            "2. If verification failed → use action: 'delegate' to an agent to fix the issues.\n"
            "3. If more tools are needed → use action: 'tool'.\n"
            "Return ONLY valid JSON in the required format."
        )

    return prompt_messages
