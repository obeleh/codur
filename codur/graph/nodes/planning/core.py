"""Planning orchestrator."""

from __future__ import annotations

import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.utils import _normalize_messages
from codur.graph.nodes.non_llm_tools import run_non_llm_tools
from codur.utils.retry import LLMRetryStrategy

from .decision_handler import PlanningDecisionHandler
from .prompt_builder import PlanningPromptBuilder
from .validators import looks_like_change_request, mentions_file_path, has_mutation_tool
from .json_parser import JSONResponseParser
from .classifier import (
    quick_classify,
    TaskType,
    ClassificationResult,
    get_agent_for_task_type,
    build_context_aware_prompt,
)

console = Console()


class PlanningOrchestrator:
    def __init__(self, config: CodurConfig) -> None:
        self.config = config
        self.prompt_builder = PlanningPromptBuilder(config)
        self.decision_handler = PlanningDecisionHandler(config)
        self.json_parser = JSONResponseParser()

    def plan(self, state: AgentState, llm: BaseChatModel) -> PlanNodeResult:
        if "config" not in state:
            raise ValueError("AgentState must include config")
        if state.get("verbose"):
            console.print("[bold blue]Planning...[/bold blue]")

        messages = _normalize_messages(state.get("messages"))
        iterations = state.get("iterations", 0)

        tool_results_present = any(
            isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
            for msg in messages
        )

        # Fast path: non-LLM tool detection
        if self.config.runtime.detect_tool_calls_from_text:
            non_llm_result = run_non_llm_tools(messages, state)
            if non_llm_result:
                return non_llm_result

        # === PHASE 1: Quick Classification ===
        # Try to classify without LLM call for obvious cases
        classification = quick_classify(messages, self.config)

        if state.get("verbose"):
            console.print(f"[dim]Phase 1: {classification.task_type.value} "
                         f"(confidence: {classification.confidence:.0%})[/dim]")

        # Handle high-confidence classifications without LLM
        if classification.is_confident and not tool_results_present:
            user_message = messages[-1].content if messages else ""
            phase1_result = self._handle_confident_classification(classification, iterations, user_message)
            if phase1_result:
                if state.get("verbose"):
                    console.print("[green]✓ Phase 1 resolved (no LLM needed)[/green]")
                return phase1_result

        # === PHASE 2: Context-Aware LLM Planning ===
        # Use a focused prompt based on Phase 1 classification
        if state.get("verbose"):
            console.print("[dim]Phase 2: Using context-aware LLM planning[/dim]")

        prompt_messages = self._build_phase2_messages(
            messages, tool_results_present, classification
        )

        retry_strategy = LLMRetryStrategy(
            max_attempts=self.config.planning.max_retry_attempts,
            initial_delay=self.config.planning.retry_initial_delay,
            backoff_factor=self.config.planning.retry_backoff_factor,
        )
        try:
            # Create LLM for planning with lower temperature for more deterministic JSON output
            from codur.llm import create_llm_profile
            planning_llm = create_llm_profile(
                self.config,
                self.config.llm.default_profile,
                json_mode=True
            )
            # Override temperature for planning (lower = more deterministic)
            if hasattr(planning_llm, 'temperature'):
                planning_llm.temperature = self.config.llm.planning_temperature

            active_llm, response, profile_name = retry_strategy.invoke_with_fallbacks(
                self.config,
                planning_llm,
                prompt_messages,
            )
            content = response.content
        except Exception as exc:
            if not self.config.agents.preferences.default_agent:
                raise ValueError("agents.preferences.default_agent must be configured") from exc
            default_agent = self.config.agents.preferences.default_agent
            if state.get("verbose"):
                console.print(f"[red]Planning failed: {str(exc)}[/red]")
                console.print(f"[yellow]Falling back to default agent: {default_agent}[/yellow]")
            return {
                "next_action": "delegate",
                "selected_agent": default_agent,
                "iterations": iterations + 1,
                "llm_debug": {"error": str(exc), "llm_profile": self.config.llm.default_profile},
            }

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
                        retry_response = active_llm.invoke([retry_prompt] + list(messages))
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
                            return {
                                "next_action": "end",
                                "final_response": content,
                                "iterations": iterations + 1,
                                "llm_debug": llm_debug,
                            }
                    else:
                        return {
                            "next_action": "end",
                            "final_response": content,
                            "iterations": iterations + 1,
                            "llm_debug": llm_debug,
                        }
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
                    if decision.get("action") != "tool":
                        retry_prompt = SystemMessage(
                            content=(
                                "You must return action 'tool' with tool_calls to edit the referenced file in JSON format. "
                                "Do not respond with instructions or summaries. Return valid JSON only."
                            )
                        )
                        retry_response = active_llm.invoke([retry_prompt] + list(messages))
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
                        retry_response = active_llm.invoke([retry_prompt] + list(messages))
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

            return self.decision_handler.handle_decision(decision, iterations, llm_debug)

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

            return {
                "next_action": "delegate",
                "selected_agent": default_agent,
                "iterations": iterations + 1,
                "llm_debug": llm_debug,
            }

    @staticmethod
    def _last_human_message(messages: list[BaseMessage]) -> str | None:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None

    def _handle_confident_classification(
        self,
        classification: ClassificationResult,
        iterations: int,
        user_message: str = ""
    ) -> PlanNodeResult | None:
        """Handle high-confidence Phase 1 classifications without LLM.

        Returns a PlanNodeResult if we can resolve without LLM, None otherwise.
        """
        task_type = classification.task_type

        # Greetings - respond directly
        if task_type == TaskType.GREETING:
            return {
                "next_action": "end",
                "final_response": "Hello! How can I help you with your coding tasks today?",
                "iterations": iterations + 1,
                "llm_debug": {"phase1_resolved": True, "task_type": "greeting"},
            }

        # File operations - generate tool calls
        if task_type == TaskType.FILE_OPERATION and classification.detected_files:
            action = classification.detected_action
            if action and classification.detected_files:
                tool_call = {"tool": action, "args": {"path": classification.detected_files[0]}}
                # Handle move/copy which need source and destination
                if action in ("move_file", "copy_file") and len(classification.detected_files) >= 2:
                    tool_call = {
                        "tool": action,
                        "args": {
                            "source": classification.detected_files[0],
                            "destination": classification.detected_files[1]
                        }
                    }
                return {
                    "next_action": "tool",
                    "tool_calls": [tool_call],
                    "iterations": iterations + 1,
                    "llm_debug": {"phase1_resolved": True, "task_type": "file_operation"},
                }

        # Explanation requests - read file first
        if task_type == TaskType.EXPLANATION and classification.detected_files:
            return {
                "next_action": "tool",
                "tool_calls": [{"tool": "read_file", "args": {"path": classification.detected_files[0]}}],
                "iterations": iterations + 1,
                "llm_debug": {"phase1_resolved": True, "task_type": "explanation"},
            }

        # Code fix/generation - delegate to appropriate agent
        if task_type in (TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.COMPLEX_REFACTOR):
            agent = get_agent_for_task_type(
                task_type,
                self.config,
                classification.detected_files,
                user_message
            )
            return {
                "next_action": "delegate",
                "selected_agent": agent,
                "iterations": iterations + 1,
                "llm_debug": {
                    "phase1_resolved": True,
                    "task_type": task_type.value,
                    "selected_agent": agent
                },
            }

        # Cannot resolve in Phase 1
        return None

    def _build_phase2_messages(
        self,
        messages: list[BaseMessage],
        has_tool_results: bool,
        classification: ClassificationResult
    ) -> list[BaseMessage]:
        """Build context-aware prompt messages for Phase 2.

        Uses classification from Phase 1 to create a focused prompt,
        reducing token usage compared to the full generic prompt.
        """
        # Build context-aware system prompt based on classification
        context_prompt = build_context_aware_prompt(classification, self.config)

        system_message = SystemMessage(content=context_prompt)
        prompt_messages = [system_message] + list(messages)

        if has_tool_results:
            followup_prompt = SystemMessage(
                content=(
                    "Tool results are available above. Use them to complete the task. "
                    "Respond with valid JSON only."
                )
            )
            prompt_messages.insert(1, followup_prompt)

        return prompt_messages
