"""Verification response tool for structured verification results."""

from dataclasses import dataclass

from codur.constants import TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import tool_scenarios


@dataclass
class VerificationResult:
    """Structured verification result."""
    passed: bool
    reasoning: str
    expected: str | None = None
    actual: str | None = None
    suggestions: str | None = None

    @property
    def status(self) -> str:
        """Get verification status as string."""
        return "PASS" if self.passed else "FAIL"

    def __str__(self) -> str:
        """Return string representation."""
        return f"Verification response recorded: {self.status}"


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
) -> VerificationResult:
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
        VerificationResult with all verification data
    """
    return VerificationResult(
        passed=passed,
        reasoning=reasoning,
        expected=expected,
        actual=actual,
        suggestions=suggestions,
    )
