"""Routing helpers for graph flow control."""

from codur.graph.state import AgentState
from codur.constants import (
    DEFAULT_MAX_ITERATIONS,
    AGENT_CODING,
    AGENT_EXPLAINING,
    REF_AGENT_CODING,
    REF_AGENT_EXPLAINING,
    ACTION_DELEGATE,
    ACTION_CONTINUE,
    ACTION_END,
)

def should_delegate(state: AgentState) -> str:
    """Decide if we should delegate to an agent or end.

    Args:
        state: Current agent state with next_action

    Returns:
        "delegate", "tool", "coding", "explaining", or "end" based on the next_action in state
    """
    next_action = state.get("next_action", ACTION_DELEGATE)
    selected_agent = state.get("selected_agent")
    if next_action == ACTION_DELEGATE:
        if selected_agent in (REF_AGENT_CODING, AGENT_CODING):
            return "coding"
        if selected_agent in (REF_AGENT_EXPLAINING, AGENT_EXPLAINING):
            return "explaining"
    return next_action if next_action != ACTION_END else ACTION_END


def should_continue(state: AgentState) -> str:
    """Decide if we should continue iterating or end.

    Args:
        state: Current agent state with iterations count

    Returns:
        "continue" or "end" based on iteration limit and next_action
    """
    iterations = state.get("iterations", 0)
    next_action = state.get("next_action", ACTION_END)

    # Get max iterations from config if available, otherwise use constant
    config = state.get("config")
    max_iterations = config.runtime.max_iterations if config else DEFAULT_MAX_ITERATIONS

    # Max iterations check
    if iterations >= max_iterations:
        return ACTION_END

    selected_agent = state.get("selected_agent")

    if next_action == ACTION_CONTINUE:
        if selected_agent in (REF_AGENT_CODING, AGENT_CODING):
            return "coding"
        if selected_agent in (REF_AGENT_EXPLAINING, AGENT_EXPLAINING):
            return "explaining"

    if next_action == "coding":
        return "coding"
    if next_action == "explaining":
        return "explaining"

    return next_action if next_action == ACTION_CONTINUE else ACTION_END