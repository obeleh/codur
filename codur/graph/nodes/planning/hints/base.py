"""Base strategy for Phase 1 planning hints."""

from typing import Protocol, runtime_checkable
from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
from codur.config import CodurConfig
from langchain_core.messages import BaseMessage

@runtime_checkable
class Phase1Strategy(Protocol):
    """Interface for Phase 1 planning strategies."""
    
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

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        """Build a context-aware prompt for Phase 2 planning."""
        ...
