"""Unknown/fallback task strategy."""

from langchain_core.messages import BaseMessage

from codur.graph.node_types import PlanNodeResult
from codur.graph.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from codur.graph.planning.strategies.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    build_example_line,
    format_examples,
    normalize_agent_name,
)

# Unknown strategy has empty patterns (it's the fallback)
_UNKNOWN_PATTERNS = PatternConfig()


class UnknownStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return empty patterns - unknown is the fallback when nothing matches."""
        return _UNKNOWN_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Unknown always returns zero - it's the fallback when nothing matches."""
        return ScoreContribution(score=0.0)

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

    def build_planning_prompt(
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
                },
            ),
        ]
        focus = (
            "**Task Focus: Unknown**\n"
            "- Follow the general planning rules and choose the safest action.\n"
            "- Use investigation tools if needed, then delegate_task() or task_complete().\n"
            "\n"
            "Examples:\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
