"""Meta-tools for agent control and clarification."""
from typing import TypedDict, NotRequired

from codur.constants import TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import tool_scenarios


class VerificationResult(TypedDict):
    """Structured verification result."""
    passed: bool
    reasoning: str
    expected: NotRequired[str | None]
    actual: NotRequired[str | None]
    suggestions: NotRequired[str | None]


class ClarificationResult(TypedDict):
    """Structured clarification result."""
    reasoning: str


@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.CODE_VALIDATION, TaskType.EXPLANATION, TaskType.META_TOOL)
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


@tool_scenarios(TaskType.RESULT_VERIFICATION, TaskType.META_TOOL)
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


@tool_scenarios(TaskType.META_TOOL)
def done(reasoning: str, state: "AgentState"=None) -> ClarificationResult:
    """Indicate that the agent has completed its task.

    Use this tool to signal that you have finished your work
    and are ready to provide the final response.

    Args:
        reasoning: Explanation of why the task is complete

    Returns:
        ClarificationResult with the reasoning echoed back
    """
    return ClarificationResult(reasoning=reasoning)


@tool_scenarios(TaskType.META_TOOL)
def task_complete(
    response: str,
    state: AgentState | None = None,
) -> dict[str, str]:
    """Signal that the task is complete and provide the response.

    Use this when:
    - You can answer the user's question directly (e.g., greetings, simple questions)
    - You have gathered enough information to provide a complete response
    - The task does not require code changes or delegation

    The response will be returned directly to the user.

    Args:
        response: The final answer or response to the user's request
        state: Agent state (ignored)

    Returns:
        Task completion confirmation
    """
    # Return value is not used - we intercept this tool call in the planning loop
    return {"status": "task complete", "response": response}