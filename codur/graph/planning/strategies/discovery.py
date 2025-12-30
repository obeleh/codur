"""Shared discovery logic for file-based strategies.

Provides reusable functions for the common file discovery pattern:
1. No files known -> list_files to discover
2. Tool results present -> select file from results
3. Files known -> read_file for context
"""

from rich.console import Console
from langchain_core.messages import BaseMessage

from codur.graph.node_types import PlanNodeResult
from codur.graph.planning.types import ClassificationResult
from codur.graph.planning.tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
)

console = Console()


def discover_files_if_needed(
    classification: ClassificationResult,
    tool_results_present: bool,
    messages: list[BaseMessage],
    iterations: int,
    verbose: bool = False,
    discovery_message: str = "No file hint detected - listing files",
    selection_message: str = "Selected file from tool results: {candidate}",
    context_message: str = "Reading {file_path} for context",
    selected_agent: str | None = None,
) -> PlanNodeResult | None:
    """Handle the common file discovery pattern.

    This implements the three-step discovery logic used by CodeFix,
    CodeGeneration, and Explanation strategies:

    1. If no tool results and no files -> call list_files
    2. If tool results present but no files -> select and read a file
    3. If files are known -> read the first file for context

    Args:
        classification: The classification result with detected files
        tool_results_present: Whether tool results exist in messages
        messages: The conversation messages
        iterations: Current iteration count
        verbose: Whether to print verbose messages
        discovery_message: Message to print when listing files
        selection_message: Message to print when selecting file (use {candidate})
        context_message: Message to print when reading context (use {file_path})
        selected_agent: Optional agent to route to after discovery (e.g., "agent:codur-coding")

    Returns:
        A PlanNodeResult if discovery action is needed, None otherwise
    """
    task_type = classification.task_type.value

    # 1. No file hint -> list files
    if not tool_results_present and not classification.detected_files:
        if verbose:
            console.print(f"[dim]{discovery_message}[/dim]")
        result: PlanNodeResult = {
            "next_action": "tool",
            "tool_calls": [{"tool": "list_files", "args": {}}],
            "iterations": iterations + 1,
            "llm_debug": {
                "phase1_resolved": True,
                "task_type": task_type,
                "file_discovery": "list_files",
            },
        }
        if selected_agent:
            result["selected_agent"] = selected_agent
        return result

    # 2. Tool results present (likely list_files) -> select and read file
    if tool_results_present and not classification.detected_files and not tool_results_include_read_file(messages):
        candidate = select_file_from_tool_results(messages)
        if candidate:
            if verbose:
                console.print(f"[dim]{selection_message.format(candidate=candidate)}[/dim]")
            result: PlanNodeResult = {
                "next_action": "tool",
                "tool_calls": [{"tool": "read_file", "args": {"path": candidate}}],
                "iterations": iterations + 1,
                "llm_debug": {
                    "phase1_resolved": True,
                    "task_type": task_type,
                    "file_discovery": candidate,
                },
            }
            if selected_agent:
                result["selected_agent"] = selected_agent
            return result

    # 3. High confidence with files -> read file for context
    if classification.is_confident and not tool_results_present and classification.detected_files:
        file_path = classification.detected_files[0]
        if verbose:
            console.print(f"[dim]{context_message.format(file_path=file_path)}[/dim]")
        result: PlanNodeResult = {
            "next_action": "tool",
            "tool_calls": [{"tool": "read_file", "args": {"path": file_path}}],
            "iterations": iterations + 1,
            "llm_debug": {
                "phase0_resolved": True,
                "task_type": task_type,
                "discovery_only": True,
            },
        }
        if selected_agent:
            result["selected_agent"] = selected_agent
        return result

    return None
