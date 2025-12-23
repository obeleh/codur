"""Unknown task strategy for Phase 1."""

from langchain_core.messages import BaseMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
from codur.config import CodurConfig
from codur.graph.nodes.planning.hints.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    build_example_line,
    format_examples,
    normalize_agent_name,
)

class UnknownStrategy:
    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:
        """Unknown tasks always pass to Phase 2."""
        return None

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        default_agent = normalize_agent_name(
            config.agents.preferences.default_agent,
            "agent:codur-coding",
        )
        examples = [
            build_example_line(
                "Write a sorting function",
                {
                    "action": "delegate",
                    "agent": default_agent,
                    "reasoning": "code generation request",
                    "response": None,
                    "tool_calls": [],
                },
            ),
        ]
        focus = (
            "**Task Focus: Unknown**\n"
            "- Follow the general planning rules and choose the safest action.\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
