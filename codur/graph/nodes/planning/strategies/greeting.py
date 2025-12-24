"""Greeting strategy."""

from rich.console import Console
from langchain_core.messages import BaseMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from codur.graph.nodes.planning.strategies.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    build_example_line,
    format_examples,
)

console = Console()

# Domain-specific patterns for greeting tasks
_GREETING_PATTERNS = PatternConfig(
    primary_keywords=frozenset({
        "hi", "hello", "hey", "yo", "sup", "thanks", "thank you",
        "good morning", "good afternoon", "good evening", "bye", "goodbye"
    }),
    boosting_keywords=frozenset(),  # Greetings don't need boosting
    negative_keywords=frozenset({
        "fix", "create", "write", "generate", "explain", "describe",
        "search", "find", "refactor", "move", "delete"
    }),
    file_extensions=frozenset(),  # Greetings don't involve files
)


class GreetingStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return patterns for greeting classification."""
        return _GREETING_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute greeting classification score."""
        result = ScoreContribution(score=0.0)
        patterns = self.get_patterns()

        # Check for greeting patterns
        is_greeting = text_lower in patterns.primary_keywords or (
            len(words) <= 3 and bool(words & patterns.primary_keywords)
        )

        if is_greeting:
            result.add(0.4, "greeting keyword")
            # Check if there are no other intent signals
            has_other = any(kw in text_lower for kw in patterns.negative_keywords) or bool(detected_files)
            if not has_other:
                result.add(0.3, "no other intent signals")

        return result

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

    def build_planning_prompt(
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
