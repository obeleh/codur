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
        
        # High confidence -> delegate
        if classification.is_confident and not tool_results_present:
            user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
            agent = select_agent_for_task(
                config=config,
                user_message=user_message,
                detected_files=classification.detected_files,
                routing_key="complex",
                prefer_multifile=True,
                allow_coding_agent=False,
            )
            
            if verbose:
                console.print("[green]✓ Complex refactor resolved (high confidence)[/green]")
            return {
                "next_action": "delegate",
                "selected_agent": agent,
                "iterations": iterations + 1,
                "llm_debug": {
                    "phase1_resolved": True,
                    "task_type": classification.task_type.value,
                    "selected_agent": agent
                },
            }

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
