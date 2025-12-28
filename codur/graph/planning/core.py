"""Planning orchestrator."""

from __future__ import annotations

import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.node_types import PlanNodeResult
from codur.graph.state_operations import get_iterations, get_llm_calls, get_messages, is_verbose
from codur.graph.non_llm_tools import run_non_llm_tools
from codur.llm import create_llm_profile
from codur.utils.retry import LLMRetryStrategy
from codur.utils.llm_calls import invoke_llm, LLMCallLimitExceeded
from codur.utils.llm_helpers import create_and_invoke
from codur.utils.validation import require_config

from .decision_handler import PlanningDecisionHandler
from .prompt_builder import PlanningPromptBuilder
from .validators import looks_like_change_request, mentions_file_path, has_mutation_tool
from .json_parser import JSONResponseParser
from .classifier import quick_classify
from .types import ClassificationResult
from .strategies import get_strategy_for_task
from .tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
)
from .injectors import inject_followup_tools

console = Console()


# Three-phase planning: pattern-plan → llm-pre-plan → llm-plan

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
        isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
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
            console.print("[green]✓ Pattern resolved via strategy[/green]")
        return result

    # No pattern match - pass to next phase
    if is_verbose(state):
        next_phase = "LLM pre-plan" if config.planning.use_llm_pre_plan else "full LLM planning"
        console.print(f"[dim]No patterns matched, moving to {next_phase}[/dim]")

    return {
        "next_action": "continue_to_llm_pre_plan",
        "iterations": iterations,
        "classification": classification,  # Pass to next phase for context
    }


def llm_pre_plan(state: AgentState, config: CodurConfig) -> PlanNodeResult:
    """Phase 1: LLM-based quick classification (experimental, gated by config).

    When enabled (config.planning.use_llm_pre_plan=True), uses a fast LLM
    to classify the task and suggest routing. More flexible than pattern
    matching for novel task types.

    When disabled (default), passes directly to Phase 2 for full planning.
    """
    # If LLM pre-plan is disabled, skip directly to full planning
    if not config.planning.use_llm_pre_plan:
        if is_verbose(state):
            console.print("[dim]LLM pre-plan disabled, passing to full planning[/dim]")
        return {
            "next_action": "continue_to_llm_plan",
            "iterations": get_iterations(state),
            "classification": state.get("classification"),
        }

    messages = get_messages(state)
    iterations = get_iterations(state)

    if is_verbose(state):
        console.print("[bold blue]Planning (Phase 1: LLM Classification)...[/bold blue]")

    # Get classification from Phase 0 if available
    classification = state.get("classification")

    # Get last human message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # Build lightweight classification prompt with clear JSON schema
    system_prompt = SystemMessage(content="""You are a task classifier. Analyze the user's request and classify it.

**Task Types:**
- greeting: greetings or thanks
- file_operation: move, copy, delete, rename files
- code_fix: fix bugs, debug, implement, solve
- code_generation: write new code, create functions
- code_validation: validate, lint, or verify code correctness
- documentation: write or update docs/README/markdown content
- explanation: explain, describe, summarize code
- complex_refactor: refactor, redesign, migrate
- web_search: weather, news, real-time data, search
- unknown: anything else

**Required JSON Response:**
{
  "task_type": "one of the types above",
  "confidence": 0.0-1.0,
  "detected_files": ["list", "of", "files"],
  "suggested_action": "respond|tool|delegate",
  "reasoning": "brief explanation"
}

Only use file_operation when the user explicitly asks to move/copy/delete/rename files.
If the user asks to implement/fix code in a file, classify as code_fix or code_generation.

**Examples:**

User: "Hello!"
{"task_type": "greeting", "confidence": 0.95, "detected_files": [], "suggested_action": "respond", "reasoning": "Simple greeting"}

User: "Fix the bug in main.py"
{"task_type": "code_fix", "confidence": 0.9, "detected_files": ["main.py"], "suggested_action": "delegate", "reasoning": "Bug fix request with file path"}

User: "Fix the bug in @main.py"
{"task_type": "code_fix", "confidence": 0.9, "detected_files": ["main.py"], "suggested_action": "delegate", "reasoning": "Bug fix request with @file path"}

User: "What does app.py do?"
{"task_type": "explanation", "confidence": 0.85, "detected_files": ["app.py"], "suggested_action": "tool", "reasoning": "Explanation request for specific file"}

User: "Write a sorting function"
{"task_type": "code_generation", "confidence": 0.8, "detected_files": [], "suggested_action": "delegate", "reasoning": "New code generation request"}

User: "Lint the project"
{"task_type": "code_validation", "confidence": 0.8, "detected_files": [], "suggested_action": "tool", "reasoning": "Code validation request"}

User: "Update the README with usage instructions"
{"task_type": "documentation", "confidence": 0.8, "detected_files": ["README.md"], "suggested_action": "tool", "reasoning": "Documentation update request"}

User: "Implement title case in @main.py"
{"task_type": "code_generation", "confidence": 0.85, "detected_files": ["main.py"], "suggested_action": "delegate", "reasoning": "Code generation request with @file path"}

User: "What's the weather in Paris?"
{"task_type": "web_search", "confidence": 0.9, "detected_files": [], "suggested_action": "tool", "reasoning": "Real-time weather data request"}

User: "hey fix main.py"
{"task_type": "code_fix", "confidence": 0.85, "detected_files": ["main.py"], "suggested_action": "delegate", "reasoning": "Casual greeting but primary intent is code fix"}

Respond with ONLY valid JSON matching the schema above.""")

    try:
        response = create_and_invoke(
            config,
            [system_prompt, HumanMessage(content=user_message)],
            profile_name=config.llm.default_profile,
            json_mode=True,
            temperature=0.2,  # Low temperature for deterministic classification
            invoked_by="planning.llm_pre_plan",
            state=state,
        )

        # Parse LLM response
        parser = JSONResponseParser()
        llm_result = parser.parse(response.content)
        if llm_result is None:
            raise ValueError("Failed to parse LLM pre-plan JSON response")

        if is_verbose(state):
            console.print(f"[dim]LLM classification: {llm_result.get('task_type')} "
                         f"(confidence: {llm_result.get('confidence', 0):.0%})[/dim]")

        # Check confidence
        confidence = llm_result.get("confidence", 0)
        if confidence >= 0.8:
            # High confidence - route based on suggested action
            action = llm_result.get("suggested_action", "delegate")
            if action == "respond":
                return {
                    "next_action": "end",
                    "final_response": llm_result.get("reasoning", "Task classified."),
                    "iterations": iterations + 1,
                    "llm_debug": {"phase1_llm_resolved": True},
                }
            elif action == "tool":
                # Would need to determine specific tool - for now, pass to Phase 2
                pass
            elif action == "delegate":
                # Could select agent here, but safer to pass to Phase 2
                pass

        # Pass to Phase 2 for routing decisions (Phase 1 only handles greetings)
        if is_verbose(state):
            console.print("[dim]Passing to Phase 2 for agent routing and planning[/dim]")

    except Exception as exc:
        if is_verbose(state):
            console.print(f"[yellow]LLM pre-plan failed: {exc}[/yellow]")

    # Pass to Phase 2 for full analysis
    return {
        "next_action": "continue_to_llm_plan",
        "iterations": iterations,
        "classification": classification,
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

        if is_verbose(state):
            console.print("[bold blue]Planning (Phase 2: Full LLM Planning)...[/bold blue]")

        def _with_llm_calls(result: PlanNodeResult) -> PlanNodeResult:
            result["llm_calls"] = get_llm_calls(state)
            return result

        messages = get_messages(state)
        iterations = get_iterations(state)

        tool_results_present = any(
            isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
            for msg in messages
        )

        # Get classification from Phase 1 if available, otherwise do it again
        classification = state.get("classification")
        if not classification:
            classification = quick_classify(messages, self.config)
            if is_verbose(state):
                console.print(f"[dim]Classification (re-evaluated): {classification.task_type.value} "
                             f"(confidence: {classification.confidence:.0%})[/dim]")
                if classification.candidates:
                    console.print(f"[dim]Candidates: {_format_candidates(classification.candidates)}[/dim]")

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
            if is_verbose(state):
                console.print(f"[dim]LLM content: {content}[/dim]")
        except Exception as exc:
            if isinstance(exc, LLMCallLimitExceeded):
                raise
            if "Failed to validate JSON" in str(exc):
                console.print("  PLANNING ERROR - LLM returned invalid JSON", style="red bold on yellow")

            default_agent = self.config.agents.preferences.default_agent
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
                default_agent = self.config.agents.preferences.default_agent
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

            # NEW VALIDATION: Ensure file context exists before delegating to codur-coding
            if (decision
                and decision.get("action") == "delegate"
                and decision.get("agent") == "agent:codur-coding"):

                # Check if file contents are available in messages
                has_file_contents = self._tool_results_include_read_file(messages)

                # For multi-file scenarios, check if we've read all discovered files
                discovered_files = self._extract_files_from_tool_results(messages)
                unread_files = []
                if discovered_files:
                    # Check which files have NOT been read yet
                    unread_files = [
                        f for f in discovered_files
                        if not any(f"read_file:" in msg.content and f in msg.content
                                   for msg in messages
                                   if isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:"))
                    ]

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

            default_agent = self.config.agents.preferences.default_agent
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

    @staticmethod
    def _extract_files_from_tool_results(messages: list[BaseMessage]) -> list[str]:
        """Extract file paths from list_files tool results in message history.

        Parses SystemMessage entries containing "Tool results:" to find file listings
        from list_files tool executions. Returns a list of discovered file paths.
        """
        files = []

        for msg in messages:
            if isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:"):
                # Look for list_files results
                if "list_files:" in msg.content:
                    # Parse the file list from the tool result
                    content = msg.content
                    if "list_files:" in content:
                        result_section = content.split("list_files:")[1].split("\n\n")[0]
                        # Extract filenames (simple pattern matching)
                        import re
                        file_matches = re.findall(r'(\w+\.py)', result_section)
                        files.extend(file_matches)

        return list(set(files))  # Remove duplicates

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
        # Use context-aware prompt based on classification to guide multi-step planning
        strategy = get_strategy_for_task(classification.task_type)
        planning_prompt = strategy.build_planning_prompt(classification, self.config)
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
