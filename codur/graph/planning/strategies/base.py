"""Base strategy for task-specific planning logic.

Each TaskStrategy owns the domain knowledge for one task type:
- Pattern matching and classification scoring
- Discovery behavior (Phase 0)
- Prompt building for LLM planning (Phase 2)
"""

from typing import Protocol, runtime_checkable
from codur.graph.node_types import PlanNodeResult
from codur.graph.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from langchain_core.messages import BaseMessage


@runtime_checkable
class TaskStrategy(Protocol):
    """Interface for task-specific planning strategies.

    Each strategy handles one TaskType and provides:
    - get_patterns(): Domain-specific patterns for classification
    - compute_score(): Calculate classification score for this task type
    - execute(): Discovery and resolution logic (Phase 0)
    - build_planning_prompt(): Context-aware prompt for LLM planning (Phase 2)
    """

    def get_patterns(self) -> PatternConfig:
        """Return patterns this strategy matches for classification."""
        ...

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute classification score for this task type.

        Args:
            text_lower: Lowercased user message
            words: Set of words from the message
            detected_files: List of file paths extracted from message
            has_code_file: Whether any detected file is a code file

        Returns:
            ScoreContribution with score and reasoning
        """
        ...

    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:
        """Execute the strategy. Returns a PlanNodeResult if resolved, None otherwise."""
        ...

    def build_planning_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        """Build a context-aware prompt for LLM planning."""
        ...


# Backward compatibility alias
Phase1Strategy = TaskStrategy
