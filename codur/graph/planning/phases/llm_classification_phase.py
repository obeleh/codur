"""Phase 1: LLM-based classification."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.constants import TaskType
from codur.graph.node_types import PlanNodeResult
from codur.graph.planning.types import ClassificationResult
from codur.graph.state import AgentState
from codur.graph.state_operations import (
    get_iterations,
    get_last_human_message_content,
    is_verbose,
)
from codur.utils.json_parser import JSONResponseParser
from codur.utils.llm_helpers import create_and_invoke

console = Console()


def _parse_classification_result(data: dict) -> ClassificationResult:
    """Parses the raw dictionary from LLM into a ClassificationResult."""
    raw_type = data.get("task_type", "").lower()

    # Handle mappings if necessary
    if raw_type == "complex_refactor":
        task_type = TaskType.REFACTOR
    else:
        try:
            task_type = TaskType(raw_type)
        except ValueError:
            task_type = TaskType.UNKNOWN

    return ClassificationResult(
        task_type=task_type,
        confidence=float(data.get("confidence", 0.0)),
        detected_files=data.get("detected_files", []),
        detected_action=data.get("suggested_action"),
        reasoning=data.get("reasoning", ""),
    )


def llm_classification(state: AgentState, config: CodurConfig) -> PlanNodeResult:
    """Phase 1: LLM-based quick classification (experimental, gated by config).

    When enabled (config.planning.use_llm_pre_plan=True), uses a fast LLM
    to classify the task and suggest routing. More flexible than pattern
    matching for novel task types.

    When disabled (default), passes directly to Phase 2 for full planning.
    """
    # If LLM classification is disabled, skip directly to full planning
    if not config.planning.use_llm_pre_plan:
        if is_verbose(state):
            console.print("[dim]LLM classification disabled, passing to full planning[/dim]")
        return {
            "next_action": "continue_to_llm_plan",
            "iterations": get_iterations(state),
            "classification": state.get("classification"),
            "next_step_suggestion": None,
        }

    iterations = get_iterations(state)

    if is_verbose(state):
        console.print("[bold blue]Planning (Phase 1: LLM Classification)...[/bold blue]")

    # Get classification from Phase 0 if available
    classification: ClassificationResult = state.get("classification")

    # Get last human message
    user_message = get_last_human_message_content(state) or ""

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
            invoked_by="planning.llm_classification",
            state=state,
        )

        # Parse LLM response
        parser = JSONResponseParser()
        raw_result = parser.parse(response.content)
        if raw_result is None:
            raise ValueError("Failed to parse LLM classification JSON response")

        classification = _parse_classification_result(raw_result)

        if is_verbose(state):
            console.print(f"[dim]LLM classification: {classification.task_type.value} "
                         f"(confidence: {classification.confidence:.0%})[/dim]")

        # Check confidence
        if classification.is_confident:
            # High confidence - route based on suggested action
            action = classification.detected_action or "delegate"
            if action == "respond":
                return {
                    "next_action": "end",
                    "final_response": classification.reasoning,
                    "iterations": iterations + 1,
                    "llm_debug": {"phase1_llm_resolved": True},
                    "next_step_suggestion": classification.reasoning,
                    "classification": classification,
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
            console.print(f"[yellow]LLM classification failed: {exc}[/yellow]")

    # Pass to Phase 2 for full analysis
    return {
        "next_action": "continue_to_llm_plan",
        "iterations": iterations,
        "classification": classification,
        "next_step_suggestion": None,
    }
