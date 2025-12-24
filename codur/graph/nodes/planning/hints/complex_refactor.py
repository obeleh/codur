"""Complex refactor strategy for Phase 1."""

from rich.console import Console
from langchain_core.messages import BaseMessage, HumanMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
from codur.graph.nodes.planning.agent_selection import select_agent_for_task
from codur.config import CodurConfig
from codur.graph.nodes.planning.hints.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    select_example_files,
    build_example_line,
    format_examples,
    normalize_agent_name,
)

console = Console()

class ComplexRefactorStrategy:
    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:

        # Complex refactoring is too risky to delegate from Phase 0
        # Pass to Phase 2 for careful analysis and routing
        if verbose and classification.is_confident:
            console.print("[dim]Complex refactor detected, deferring to Phase 2 for analysis[/dim]")

        return None

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        default_agent = normalize_agent_name(
            config.agents.preferences.routing.get("multifile"),
            normalize_agent_name(
                config.agents.preferences.default_agent,
                "agent:codur-coding",
            ),
        )
        first_file, second_file = select_example_files(classification.detected_files)
        examples = [
            build_example_line(
                "Refactor the project to use a service layer",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "discover files for multi-file refactor",
                    "response": None,
                    "tool_calls": [{"tool": "list_files", "args": {}}],
                },
            ),
            build_example_line(
                f"Refactor {first_file} and {second_file} to share a base class",
                {
                    "action": "delegate",
                    "agent": default_agent,
                    "reasoning": "multi-file refactor",
                    "response": None,
                    "tool_calls": [],
                },
            ),
        ]
        focus = (
            "**Task Focus: Complex Refactor**\n"
            "- This likely spans multiple files; consider list_files if no hints are present.\n"
            "- Prefer multi-file capable agents when routing.\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
