"""Clarification tool for LLM self-explanation."""

from typing import TypedDict

from codur.constants import TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import tool_scenarios


class ClarificationResult(TypedDict):
    """Structured clarification result."""
    reasoning: str


@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.CODE_VALIDATION, TaskType.EXPLANATION)
def clarify(
    reasoning: str,
    state: AgentState | None = None,
) -> ClarificationResult:
    """Allow the LLM to explain its reasoning or thought process.

    Use this tool to communicate your reasoning, clarify ambiguity,
    or explain why you're taking a particular approach.

    Args:
        reasoning: Explanation of your reasoning or thought process
        state: Agent state (ignored)

    Returns:
        ClarificationResult with the reasoning echoed back
    """
    return ClarificationResult(reasoning=reasoning)