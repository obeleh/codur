"""Planning orchestrator."""

from __future__ import annotations

import ast
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.utils import _normalize_messages
from codur.graph.nodes.non_llm_tools import run_non_llm_tools
from codur.llm import create_llm_profile
from codur.utils.retry import LLMRetryStrategy
from codur.utils.llm_calls import invoke_llm, LLMCallLimitExceeded

from .decision_handler import PlanningDecisionHandler
from .prompt_builder import PlanningPromptBuilder
from .validators import looks_like_change_request, mentions_file_path, has_mutation_tool
from .json_parser import JSONResponseParser
from .classifier import quick_classify
from .types import ClassificationResult
from .hints import get_strategy_for_task
from .tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
)

console = Console()


# Three-phase planning: textual-pre-plan → llm-pre-plan → llm-plan

def textual_pre_plan(state: AgentState, config: CodurConfig) -> PlanNodeResult:
    """Phase 0: Textual (pattern-based) pre-planning without LLM.

    Fast path for obvious cases detected by pattern matching:
    - Greetings
    - File operations
    - Obvious non-coding requests

    If no match, passes to llm-pre-plan.
    """
    messages = _normalize_messages(state.get("messages"))
    iterations = state.get("iterations", 0)

    if state.get("verbose"):
        console.print("[bold blue]Planning (Phase 0: Textual)...[/bold blue]")

    # Fast path: non-LLM tool detection
    if config.runtime.detect_tool_calls_from_text:
        non_llm_result = run_non_llm_tools(messages, state)
        if non_llm_result:
            if state.get("verbose"):
                console.print("[green]✓ Textual pre-plan resolved[/green]")
            return non_llm_result

    # No match - pass to llm-pre-plan
    if state.get("verbose"):
        console.print("[dim]No textual patterns matched, moving to llm-pre-plan[/dim]")

    return {
        "next_action": "continue_to_llm_pre_plan",
        "iterations": iterations,
    }


def llm_pre_plan(state: AgentState, config: CodurConfig) -> PlanNodeResult:
    """Phase 1: LLM-based quick classification.

    Fast classification for obvious cases without full LLM planning:
    - Greetings
    - Simple file operations
    - Clear code fix/generation tasks (90%+ confidence)

    If confident, returns routing decision directly.
    If uncertain, passes to llm-plan for full LLM analysis.
    """
    messages = _normalize_messages(state.get("messages"))
    iterations = state.get("iterations", 0)

    if state.get("verbose"):
        console.print("[bold blue]Planning (Phase 1: LLM Pre-Plan)...[/bold blue]")

    tool_results_present = any(
        isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
        for msg in messages
    )

    # Quick classification
    classification = quick_classify(messages, config)

    if state.get("verbose"):
        console.print(f"[dim]Classification: {classification.task_type.value} "
                     f"(confidence: {classification.confidence:.0%})[/dim]")

    # Use task-specific strategy for hints or direct resolution
    strategy = get_strategy_for_task(classification.task_type)
    result = strategy.execute(
        classification,
        tool_results_present,
        messages,
        iterations,
        config,
        verbose=state.get("verbose", False)
    )
    
    if result:
        return result

    # Confidence not high enough or strategy passed - pass to llm-plan for full analysis
    if state.get("verbose"):
        console.print("[dim]Confidence insufficient or strategy passed, moving to full LLM planning[/dim]")

    return {
        "next_action": "continue_to_llm_plan",
        "iterations": iterations,
        "classification": classification,  # Pass classification to llm-plan for context
    }


class PlanningOrchestrator:
    def __init__(self, config: CodurConfig) -> None:
        self.config = config
        self.prompt_builder = PlanningPromptBuilder(config)
        self.decision_handler = PlanningDecisionHandler(config)
        self.json_parser = JSONResponseParser()

    def llm_plan(self, state: AgentState, llm: BaseChatModel) -> PlanNodeResult:
        """Phase 2: Full LLM planning for uncertain cases.

        This is the main planning phase that handles cases where:
        - Textual patterns didn't match (Phase 0)
        - Classification confidence was insufficient (Phase 1)

        Uses context-aware prompt based on Phase 1 classification to reduce token usage.
        """
        if "config" not in state:
            raise ValueError("AgentState must include config")

        if state.get("verbose"):
            console.print("[bold blue]Planning (Phase 2: Full LLM Planning)...[/bold blue]")

        def _with_llm_calls(result: PlanNodeResult) -> PlanNodeResult:
            result["llm_calls"] = state.get("llm_calls", 0)
            return result

        messages = _normalize_messages(state.get("messages"))
        iterations = state.get("iterations", 0)

        tool_results_present = any(
            isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
            for msg in messages
        )

        # Get classification from Phase 1 if available, otherwise do it again
        classification = state.get("classification")
        if not classification:
            classification = quick_classify(messages, self.config)
            if state.get("verbose"):
                console.print(f"[dim]Classification (re-evaluated): {classification.task_type.value} "
                             f"(confidence: {classification.confidence:.0%})[/dim]")

        prompt_messages = self._build_phase2_messages(
            messages, tool_results_present, classification
        )

        # If list_files results are present, select a likely python file and read it.
        if (
            tool_results_present
            and not classification.detected_files
            and not self._tool_results_include_read_file(messages)
        ):
            candidate = self._select_file_from_tool_results(messages)
            if candidate:
                return _with_llm_calls({
                    "next_action": "tool",
                    "tool_calls": [{"tool": "read_file", "args": {"path": candidate}}],
                    "iterations": iterations + 1,
                    "llm_debug": {
                        "phase2_resolved": True,
                        "task_type": classification.task_type.value,
                        "file_discovery": candidate,
                    },
                })

        # If no file hint is available for a change request, list files for discovery.
        last_human_msg = self._last_human_message(messages)
        if (
            not tool_results_present
            and not classification.detected_files
            and last_human_msg
            and looks_like_change_request(last_human_msg)
        ):
            return _with_llm_calls({
                "next_action": "tool",
                "tool_calls": [{"tool": "list_files", "args": {}}],
                "iterations": iterations + 1,
                "llm_debug": {
                    "phase2_resolved": True,
                    "task_type": classification.task_type.value,
                    "file_discovery": "list_files",
                },
            })

        retry_strategy = LLMRetryStrategy(
            max_attempts=self.config.planning.max_retry_attempts,
            initial_delay=self.config.planning.retry_initial_delay,
            backoff_factor=self.config.planning.retry_backoff_factor,
        )
        try:
            # Create LLM for planning with lower temperature for more deterministic JSON output
            planning_llm = create_llm_profile(
                self.config,
                self.config.llm.default_profile,
                json_mode=True,
                temperature=self.config.llm.planning_temperature
            )

            active_llm, response, profile_name = retry_strategy.invoke_with_fallbacks(
                self.config,
                planning_llm,
                prompt_messages,
                state=state,
                invoked_by="planning.llm_plan",
            )
            content = response.content
            if state.get("verbose"):
                console.print(f"[dim]LLM content: {content}[/dim]")
        except Exception as exc:
            if isinstance(exc, LLMCallLimitExceeded):
                raise
            if "Failed to validate JSON" in str(exc):
                console.print("  PLANNING ERROR - LLM returned invalid JSON", style="red bold on yellow")

            if not self.config.agents.preferences.default_agent:
                raise ValueError("agents.preferences.default_agent must be configured") from exc
            default_agent = self.config.agents.preferences.default_agent
            if state.get("verbose"):
                console.print(f"[red]Planning failed: {str(exc)}[/red]")
                console.print(f"[yellow]Falling back to default agent: {default_agent}[/yellow]")
            return _with_llm_calls({
                "next_action": "delegate",
                "selected_agent": default_agent,
                "iterations": iterations + 1,
                "llm_debug": {"error": str(exc), "llm_profile": self.config.llm.default_profile},
            })

        user_message = messages[-1].content if messages else ""
        planning_prompt = self.prompt_builder.build_system_prompt()
        llm_debug = self.decision_handler.create_llm_debug(
            planning_prompt,
            user_message,
            content,
            llm=active_llm,
        )
        llm_debug["llm_profile"] = profile_name

        try:
            decision = self.decision_handler.parse_planning_response(
                active_llm,
                content,
                messages,
                tool_results_present,
                llm_debug,
                state=state,
            )

            if decision is None:
                if tool_results_present:
                    last_human_msg = self._last_human_message(messages)
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
                            config=self.config,
                        )
                        retry_content = retry_response.content
                        debug_long = self.config.planning.debug_truncate_long
                        llm_debug["llm_response_retry_forced"] = (
                            retry_content[:debug_long] + "..."
                            if len(retry_content) > debug_long
                            else retry_content
                        )
                        forced_decision = self.json_parser.parse(retry_content)
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
                if not self.config.agents.preferences.default_agent:
                    raise ValueError("agents.preferences.default_agent must be configured")
                default_agent = self.config.agents.preferences.default_agent
                decision = {
                    "action": "delegate",
                    "agent": default_agent,
                    "reasoning": "No clear decision",
                    "response": None,
                }

            if decision is not None and tool_results_present:
                last_human_msg = self._last_human_message(messages)
                if last_human_msg and looks_like_change_request(last_human_msg) and mentions_file_path(last_human_msg):
                    allow_delegate = (
                        decision.get("action") == "delegate"
                        and decision.get("agent") == "agent:codur-coding"
                        and self._tool_results_include_read_file(messages)
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
                            config=self.config,
                        )
                        retry_content = retry_response.content
                        debug_long = self.config.planning.debug_truncate_long
                        llm_debug["llm_response_retry_forced"] = (
                            retry_content[:debug_long] + "..."
                            if len(retry_content) > debug_long
                            else retry_content
                        )
                        forced_decision = self.json_parser.parse(retry_content)
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
                            config=self.config,
                        )
                        retry_content = retry_response.content
                        debug_long = self.config.planning.debug_truncate_long
                        llm_debug["llm_response_retry_mutation"] = (
                            retry_content[:debug_long] + "..."
                            if len(retry_content) > debug_long
                            else retry_content
                        )
                        forced_decision = self.json_parser.parse(retry_content)
                        if forced_decision:
                            decision = forced_decision

            return _with_llm_calls(self.decision_handler.handle_decision(decision, iterations, llm_debug))

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

            if not self.config.agents.preferences.default_agent:
                raise ValueError("agents.preferences.default_agent must be configured")
            default_agent = self.config.agents.preferences.default_agent
            console.print(f"  Falling back to default agent: {default_agent}", style="yellow")

            return _with_llm_calls({
                "next_action": "delegate",
                "selected_agent": default_agent,
                "iterations": iterations + 1,
                "llm_debug": llm_debug,
            })

    @staticmethod
    def _last_human_message(messages: list[BaseMessage]) -> str | None:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None

    @staticmethod
    def _tool_results_include_read_file(messages: list[BaseMessage]) -> bool:
        return tool_results_include_read_file(messages)

    @staticmethod
    def _select_file_from_tool_results(messages: list[BaseMessage]) -> str | None:
        return select_file_from_tool_results(messages)

    def _build_phase2_messages(
        self,
        messages: list[BaseMessage],
        has_tool_results: bool,
        classification: ClassificationResult
    ) -> list[BaseMessage]:
        """Build prompt messages for Phase 2 with context-aware planning guidance.

        Uses the classification from Phase 1 to build a focused, task-specific prompt
        that guides the LLM through multi-step reasoning (Chain-of-Thought) for
        file-based coding tasks.
        """
        # Use context-aware prompt based on Phase 1 classification to guide multi-step planning
        strategy = get_strategy_for_task(classification.task_type)
        planning_prompt = strategy.build_phase2_prompt(classification, self.config)
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
