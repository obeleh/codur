"""Verification response tool for structured verification results."""

from typing import Any

from codur.constants import TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import tool_scenarios


@tool_scenarios(TaskType.RESULT_VERIFICATION)
def build_verification_response(
    passed: bool,
    reasoning: str,
    expected: str | None = None,
    actual: str | None = None,
    suggestions: str | None = None,
    root: str | None = None,
    state: AgentState | None = None,
    allow_outside_root: bool = False,
) -> str:
    """Build verification response with structured data.

    This tool doesn't execute anything - it captures the verification decision.
    The verification agent calls this as the final step after analyzing results.

    Args:
        passed: True if verification passed, False if failed
        reasoning: Explanation of the verification decision with evidence
        expected: What was expected (for FAIL cases)
        actual: What was actually observed (for FAIL cases)
        suggestions: Specific actionable advice to fix issues (for FAIL cases)
        root: Project root (ignored)
        state: Agent state (ignored)
        allow_outside_root: Permission flag (ignored)

    Returns:
        Confirmation message
    """
    status = "PASS" if passed else "FAIL"
    return f"Verification response recorded: {status}"
