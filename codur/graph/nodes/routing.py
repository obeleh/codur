"""Routing helpers for graph flow control."""

from codur.graph.state import AgentState

# Constants
MAX_ITERATIONS = 10


def should_delegate(state: AgentState) -> str:
    """Decide if we should delegate to an agent or end.

    Args:
        state: Current agent state with next_action

    Returns:
        "delegate", "tool", or "end" based on the next_action in state
    """
    next_action = state.get("next_action", "delegate")
    return next_action if next_action != "end" else "end"


def should_continue(state: AgentState) -> str:
    """Decide if we should continue iterating or end.

    Args:
        state: Current agent state with iterations count

    Returns:
        "continue" or "end" based on iteration limit and next_action
    """
    iterations = state.get("iterations", 0)
    next_action = state.get("next_action", "end")

    # Max iterations check
    if iterations >= MAX_ITERATIONS:
        return "end"

    return next_action if next_action == "continue" else "end"
