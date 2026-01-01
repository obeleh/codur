"""Phase 2: Full LLM planning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.constants import TaskType
from codur.graph.node_types import PlanNodeResult
from codur.graph.state import AgentState
from codur.graph.state_operations import (
    get_iterations,
    get_llm_calls,
    get_messages,
    is_verbose,
    get_first_human_message_content_from_messages
)
from codur.graph.utils import (
    extract_list_files_output,
    extract_read_file_paths,
)
from codur.llm import create_llm_profile
from codur.utils.retry import LLMRetryStrategy
from codur.utils.llm_calls import invoke_llm, LLMCallLimitExceeded
from codur.utils.validation import require_config
from codur.graph.planning.classifier import quick_classify
from codur.graph.planning.types import ClassificationResult
from codur.graph.planning.strategies import get_strategy_for_task
from codur.graph.planning.tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
)
from codur.graph.planning.validators import (
    has_mutation_tool,
    looks_like_change_request,
    mentions_file_path,
)
from codur.graph.planning.injectors import inject_followup_tools

if TYPE_CHECKING:
    from codur.graph.planning.prompt_builder import PlanningPromptBuilder
    from codur.graph.planning.decision_handler import PlanningDecisionHandler
    from codur.utils.json_parser import JSONResponseParser

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

    # Get classification from Phase 1 if available, otherwise do it again
    classification = state.get("classification")
    if not classification:
        classification = quick_classify(messages, config)
        if is_verbose(state):
            console.print(f"[dim]Classification (re-evaluated): {classification.task_type.value} "
                         f"(confidence: {classification.confidence:.0%})[/dim]")
            if classification.candidates:
                console.print(f"[dim]Candidates: {_format_candidates(classification.candidates)}[/dim]")

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
                "llm_debug": {
                    "phase2_resolved": True,
                    "task_type": classification.task_type.value,
                    "file_discovery": candidate,
                },
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
            "llm_debug": {
                "phase2_resolved": True,
                "task_type": classification.task_type.value,
                "file_discovery": "list_files",
            },
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

        default_agent = config.agents.preferences.default_agent
        require_config(
            default_agent,
            "agents.preferences.default_agent",
            "agents.preferences.default_agent must be configured",
        )
        if is_verbose(state):
            console.print(f"[red]Planning failed: {str(exc)}[/red]")
            console.print(f"[yellow]Falling back to default agent: {default_agent}[/yellow]")
        return _with_llm_calls({
            "next_action": "delegate",
            "selected_agent": default_agent,
            "iterations": iterations + 1,
            "llm_debug": {"error": str(exc), "llm_profile": config.llm.default_profile},
        })

    user_message = messages[-1].content if messages else ""
    planning_prompt = prompt_builder.build_system_prompt()
    llm_debug = decision_handler.create_llm_debug(
        planning_prompt,
        user_message,
        content,
        llm=active_llm,
    )
    llm_debug["llm_profile"] = profile_name

    try:
        decision = decision_handler.parse_planning_response(
            active_llm,
            content,
            messages,
            tool_results_present,
            llm_debug,
            state=state,
        )

        if decision is None:
            if tool_results_present:
                last_human_msg = get_last_human_message(messages)
                if last_human_msg and looks_like_change_request(last_human_msg) and mentions_file_path(last_human_msg):
                    retry_prompt = SystemMessage(
                        content=(
                            "You must return ONLY a valid JSON object with action 'tool' and tool_calls "
                            "that modify the referenced file."
                        )
                    )
                    retry_response = invoke_llm(
                        active_llm,
                        [retry_prompt] + list(messages),
                        invoked_by="planning.retry_force_tool",
                        state=state,
                        config=config,
                    )
                    retry_content = retry_response.content
                    debug_long = config.planning.debug_truncate_long
                    llm_debug["llm_response_retry_forced"] = (
                        retry_content[:debug_long] + "..."
                        if len(retry_content) > debug_long
                        else retry_content
                    )
                    forced_decision = json_parser.parse(retry_content)
                    if forced_decision:
                        decision = forced_decision
                    else:
                        return _with_llm_calls({
                            "next_action": "end",
                            "final_response": content,
                            "iterations": iterations + 1,
                            "llm_debug": llm_debug,
                        })
                else:
                    return _with_llm_calls({
                        "next_action": "end",
                        "final_response": content,
                        "iterations": iterations + 1,
                        "llm_debug": llm_debug,
                    })
            default_agent = config.agents.preferences.default_agent
            require_config(
                default_agent,
                "agents.preferences.default_agent",
                "agents.preferences.default_agent must be configured",
            )
            decision = {
                "action": "delegate",
                "agent": default_agent,
                "reasoning": "No clear decision",
                "response": None,
            }

        if decision is not None and tool_results_present:
            last_human_msg = get_last_human_message(messages)
            if last_human_msg and looks_like_change_request(last_human_msg) and mentions_file_path(last_human_msg):
                allow_delegate = (
                    decision.get("action") == "delegate"
                    and decision.get("agent") == "agent:codur-coding"
                    and tool_results_include_read_file(messages)
                )
                if decision.get("action") != "tool" and not allow_delegate:
                    retry_prompt = SystemMessage(
                        content=(
                            "You must return action 'tool' with tool_calls to edit the referenced file in JSON format. "
                            "Do not respond with instructions or summaries. Return valid JSON only."
                        )
                    )
                    retry_response = invoke_llm(
                        active_llm,
                        [retry_prompt] + list(messages),
                        invoked_by="planning.retry_force_tool",
                        state=state,
                        config=config,
                    )
                    retry_content = retry_response.content
                    debug_long = config.planning.debug_truncate_long
                    llm_debug["llm_response_retry_forced"] = (
                        retry_content[:debug_long] + "..."
                        if len(retry_content) > debug_long
                        else retry_content
                    )
                    forced_decision = json_parser.parse(retry_content)
                    if forced_decision:
                        decision = forced_decision
                elif not has_mutation_tool(decision.get("tool_calls", [])):
                    retry_prompt = SystemMessage(
                        content=(
                            "You must include at least one file-modification tool call in JSON format "
                            "(e.g., replace_in_file, write_file, append_file, set_*_value). "
                            "Return a valid JSON object."
                        )
                    )
                    retry_response = invoke_llm(
                        active_llm,
                        [retry_prompt] + list(messages),
                        invoked_by="planning.retry_force_mutation",
                        state=state,
                        config=config,
                    )
                    retry_content = retry_response.content
                    debug_long = config.planning.debug_truncate_long
                    llm_debug["llm_response_retry_mutation"] = (
                        retry_content[:debug_long] + "..."
                        if len(retry_content) > debug_long
                        else retry_content
                    )
                    forced_decision = json_parser.parse(retry_content)
                    if forced_decision:
                        decision = forced_decision

        # NEW VALIDATION: Ensure file context exists before delegating to codur-coding
        if (decision
            and decision.get("action") == "delegate"
            and decision.get("agent") == "agent:codur-coding"):

            # Check if file contents are available in messages
            has_file_contents = tool_results_include_read_file(messages)

            # For multi-file scenarios, check if we've read all discovered files
            discovered_files = extract_list_files_output(messages)
            unread_files = []
            if discovered_files:
                # Check which files have NOT been read yet
                read_paths = extract_read_file_paths(messages)
                unread_files = [f for f in discovered_files if f not in read_paths]

            if not has_file_contents:
                # No file context yet - need to read files first

                if discovered_files:
                    # We have a list of files - read them
                    if unread_files:
                        files_to_read = unread_files

                        tool_calls = [
                            {"tool": "read_file", "args": {"path": f}}
                            for f in files_to_read
                        ]

                        # Inject language-specific tools via injector system
                        tool_calls = inject_followup_tools(tool_calls)

                        if is_verbose(state):
                            console.print(
                                f"[yellow]Intercepted delegate to codur-coding without file context. "
                                f"Reading {files_to_read} first...[/yellow]"
                            )

                        # Convert decision to tool action
                        decision = {
                            "action": "tool",
                            "agent": "agent:codur-coding",  # Keep agent hint for routing after read
                            "reasoning": "Reading file context before coding",
                            "tool_calls": tool_calls,
                        }
                else:
                    # No files discovered yet - list files first
                    if is_verbose(state):
                        console.print(
                            "[yellow]Intercepted delegate to codur-coding without file discovery. "
                            "Listing files first...[/yellow]"
                        )

                    decision = {
                        "action": "tool",
                        "agent": "agent:codur-coding",  # Keep agent hint
                        "reasoning": "Discovering files before reading for coding context",
                        "tool_calls": [{"tool": "list_files", "args": {}}],
                    }
            elif unread_files:
                # Some files have been read but others haven't - read the remaining ones
                if is_verbose(state):
                    console.print(
                        f"[yellow]Intercepted delegate to codur-coding with partial file context. "
                        f"Reading remaining files: {unread_files}...[/yellow]"
                    )

                tool_calls = [
                    {"tool": "read_file", "args": {"path": f}}
                    for f in unread_files
                ]

                # Inject language-specific tools via injector system
                tool_calls = inject_followup_tools(tool_calls)

                # Convert decision to tool action
                decision = {
                    "action": "tool",
                    "agent": "agent:codur-coding",  # Keep agent hint
                    "reasoning": "Reading remaining file context before coding",
                    "tool_calls": tool_calls,
                }

        return _with_llm_calls(decision_handler.handle_decision(decision, iterations, llm_debug))

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        console.print("\n" + "=" * 80, style="red bold")
        console.print("  PLANNING ERROR - Failed to parse LLM decision", style="red bold on yellow")
        console.print("=" * 80, style="red bold")
        console.print(f"  Error Type: {type(exc).__name__}", style="red")
        console.print(f"  Error Message: {str(exc)}", style="red")
        console.print(f"  LLM Response: {content[:200]}...", style="yellow")
        console.print("=" * 80 + "\n", style="red bold")

        llm_debug["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "raw_response": content[:500],
        }

        default_agent = config.agents.preferences.default_agent
        require_config(
            default_agent,
            "agents.preferences.default_agent",
            "agents.preferences.default_agent must be configured",
        )
        console.print(f"  Falling back to default agent: {default_agent}", style="yellow")

        return _with_llm_calls({
            "next_action": "delegate",
            "selected_agent": default_agent,
            "iterations": iterations + 1,
            "llm_debug": llm_debug,
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
        # For retries or tool results, be explicit about next steps
        followup_prompt = SystemMessage(
            content=(
                "Tool results or verification errors are available above. Review them and respond with valid JSON.\n"
                "1. If tool results (like web search or file read) provide the answer → use action: 'respond' with the answer.\n"
                "2. If verification failed → use action: 'delegate' to an agent to fix the issues.\n"
                "3. If more tools are needed → use action: 'tool'.\n"
                "Return ONLY valid JSON in the required format."
            )
        )
        prompt_messages.insert(1, followup_prompt)

    return prompt_messages
