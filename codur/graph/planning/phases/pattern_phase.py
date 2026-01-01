"""Phase 0: Pattern-based planning."""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.node_types import PlanNodeResult
from codur.graph.state import AgentState
from codur.graph.state_operations import get_iterations, get_messages, is_verbose
from codur.graph.non_llm_tools import run_non_llm_tools
from codur.graph.planning.classifier import quick_classify
from codur.graph.planning.strategies import get_strategy_for_task

console = Console()


def _format_candidates(candidates, limit: int = 5) -> str:
    if not candidates:
        return "none"
    trimmed = candidates[:limit]
    return ", ".join(f"{item.task_type.value}:{item.confidence:.0%}" for item in trimmed)


def pattern_plan(state: AgentState, config: CodurConfig) -> PlanNodeResult:
    """Phase 0: Pattern-based pre-planning (no LLM calls).

    Combines fast pattern matching with classification-based strategies:
    1. Instant resolution for trivial cases (greetings, basic file ops)
    2. Pattern classification with task-specific routing strategies

    If resolved, routes directly. If uncertain, passes to llm-pre-plan.
    """
    messages = get_messages(state)
    iterations = get_iterations(state)

    if is_verbose(state):
        console.print("[bold blue]Planning (Phase 0: Pattern Matching)...[/bold blue]")

    # Step 1: Try instant resolution for trivial cases
    if config.runtime.detect_tool_calls_from_text:
        non_llm_result = run_non_llm_tools(messages, state)
        if non_llm_result:
            if is_verbose(state):
                console.print("[green]✓ Pattern resolved instantly[/green]")
            return non_llm_result

    # Step 2: Try classification-based strategy routing
    tool_results_present = any(
        isinstance(msg, ToolMessage)
        for msg in messages
    )

    classification = quick_classify(messages, config)

    if is_verbose(state):
        console.print(f"[dim]Classification: {classification.task_type.value} "
                     f"(confidence: {classification.confidence:.0%})[/dim]")
        if classification.candidates:
            console.print(f"[dim]Candidates: {_format_candidates(classification.candidates)}[/dim]")

    # Use task-specific strategy for hints or direct resolution
    strategy = get_strategy_for_task(classification.task_type)
    result = strategy.execute(
        classification,
        tool_results_present,
        messages,
        iterations,
        config,
        verbose=is_verbose(state)
    )

    if result:
        if is_verbose(state):
            console.print(f"[green]✓ Pattern resolved via strategy[/green] {result}")
        return result

    # No pattern match - pass to next phase
    if is_verbose(state):
        next_phase = "LLM pre-plan" if config.planning.use_llm_pre_plan else "full LLM planning"
        console.print(f"[dim]No patterns matched, moving to {next_phase}[/dim]")

    return {
        "next_action": "continue_to_llm_pre_plan",
        "iterations": iterations,
        "classification": classification,  # Pass to next phase for context
        "next_step_suggestion": None,
    }
