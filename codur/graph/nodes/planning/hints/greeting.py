"""Greeting strategy for Phase 1."""

from rich.console import Console
from langchain_core.messages import BaseMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
from codur.config import CodurConfig
from codur.graph.nodes.planning.hints.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    build_example_line,
    format_examples,
)

console = Console()

class GreetingStrategy:
    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:
        """Handle greetings directly."""
        if tool_results_present:
            return None
            
        if verbose:
            console.print("[green]✓ Greeting resolved (high confidence)[/green]")
            
        return {
            "next_action": "end",
            "final_response": "Hello! How can I help you with your coding tasks today?",
            "iterations": iterations + 1,
            "llm_debug": {"phase1_resolved": True, "task_type": "greeting"},
        }

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        examples = [
            build_example_line(
                "Hello",
                {
                    "action": "respond",
                    "agent": None,
                    "reasoning": "greeting",
                    "response": "Hello! How can I help?",
                    "tool_calls": [],
                },
            ),
        ]
        focus = (
            "**Task Focus: Greeting**\n"
            "- Use action: \"respond\" with a short friendly greeting.\n"
            "- Do not call tools or delegate.\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
