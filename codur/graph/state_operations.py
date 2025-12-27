from typing import TYPE_CHECKING, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from codur.graph.nodes.utils import normalize_messages as _normalize_messages

if TYPE_CHECKING:
    # Avoid circular imports for type checking
    # We don't want to import anything from codur.graph.state or codur.graph.nodes.types at runtime

    from codur.graph.state import AgentState
    from codur.graph.nodes.types import ExecuteNodeResult
    from codur.config import CodurConfig

# ============================================================================
# Verbose Logging Helpers
# ============================================================================

def is_verbose(state: "AgentState") -> bool:
    """Check if verbose logging is enabled in state."""
    return state.get("verbose", False)

def get_config(state: "AgentState") -> "CodurConfig | None":
    """Get the runtime configuration attached to state, if any."""
    return state.get("config")

# ============================================================================
# Message Operations
# ============================================================================

def add_message(state: "AgentState", message: BaseMessage) -> None:
    """Add a message to the state's message history."""
    messages = list(state.get("messages", []))
    state["messages"] = messages + [message]

def get_messages(state: "AgentState") -> list[BaseMessage]:
    """Get normalized messages from state."""
    messages = state.get("messages", [])
    return _normalize_messages(messages)

def get_last_human_message(state: "AgentState") -> Optional[BaseMessage]:
    """Get the last human message from state, if any."""
    messages = get_messages(state)
    if not messages:
        return None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None

def prune_messages(messages: list, max_to_keep: int = 10) -> list:
    """Prune old messages to prevent context explosion while preserving learning context.

    Keeps:
    - Original HumanMessage(s) - typically the first message
    - Recent AIMessage(s) - agent's recent attempts (so it learns from them)
    - Recent SystemMessage(s) - recent error/verification messages for context

    This helps agents learn from their mistakes by seeing their own attempts and the feedback.

    Args:
        messages: List of messages to prune
        max_to_keep: Maximum number of recent messages to keep after the first human message

    Returns:
        Pruned message list
    """
    if len(messages) <= max_to_keep:
        return messages

    # Find the first human message (original task)
    first_human_idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            first_human_idx = i
            break

    if first_human_idx is None:
        # No human message, just keep last N
        return messages[-max_to_keep:]

    # Keep original task
    pruned = messages[:first_human_idx + 1]

    # Keep recent agent attempts and error messages together (last 8 attempts)
    # This way agent sees: "I tried X, got error Y, I tried Z, got error W"
    recent_count = 0
    max_recent = 8
    for i, msg in enumerate(reversed(messages[first_human_idx + 1:])):
        # Keep both AIMessage (agent attempts) and SystemMessage (errors) from recent history
        if isinstance(msg, (AIMessage, SystemMessage)):
            if isinstance(msg, SystemMessage) and "Verification failed" not in msg.content:
                continue  # Skip non-error system messages
            pruned.append(msg)
            recent_count += 1
            if recent_count >= max_recent:
                break

    # Reverse to maintain chronological order
    if len(pruned) > first_human_idx + 1:
        recent = pruned[first_human_idx + 1:]
        recent.reverse()
        pruned = pruned[:first_human_idx + 1] + recent

    return pruned


# ============================================================================
# Iteration & Loop Control
# ============================================================================

def get_iterations(state: "AgentState") -> int:
    """Get the current iteration count."""
    return state.get("iterations", 0)

def increment_iterations(state: "AgentState") -> int:
    """Increment and return the iteration count."""
    iterations = get_iterations(state)
    iterations += 1
    state["iterations"] = iterations
    return iterations

def should_continue_iterating(state: "AgentState", max_iterations: int) -> bool:
    """Check if we should continue iterating."""
    return get_iterations(state) < max_iterations

# ============================================================================
# LLM Call Tracking
# ============================================================================

def get_llm_calls(state: "AgentState") -> int:
    """Get the current LLM call count."""
    return state.get("llm_calls", 0)

def add_llm_call(state: "AgentState", execute_result: "ExecuteNodeResult") -> None:
    """Add LLM calls from an execution result to the state."""
    llm_calls = get_llm_calls(state)
    llm_calls += execute_result.get("llm_calls", 0)
    state["llm_calls"] = llm_calls

def increment_llm_calls(state: "AgentState", count: int = 1) -> int:
    """Increment LLM call counter."""
    llm_calls = get_llm_calls(state)
    llm_calls += count
    state["llm_calls"] = llm_calls
    return llm_calls

def check_llm_call_limit(state: "AgentState") -> bool:
    """Check if we have exceeded the LLM call limit."""
    max_llm_calls = state.get("max_llm_calls")
    if max_llm_calls is None:
        return True  # No limit
    return get_llm_calls(state) < max_llm_calls

# ============================================================================
# Outcome & Status Operations
# ============================================================================

def get_outcome(execute_result: "ExecuteNodeResult") -> str:
    """Extract result from an execution outcome, raising on error."""
    outcome = execute_result.get("agent_outcome", {})
    if outcome.get("status") == "error":
        raise Exception(f"Agent execution failed: {outcome.get('result')}")
    return outcome.get("result", "")

def get_agent_outcome(state: "AgentState") -> dict:
    """Get the agent outcome from state."""
    return state.get("agent_outcome", {})

def set_agent_outcome(state: "AgentState", outcome: dict) -> None:
    """Set the agent outcome in state."""
    state["agent_outcome"] = outcome

def get_outcome_status(state: "AgentState") -> str:
    """Get the status from the agent outcome."""
    return get_agent_outcome(state).get("status", "unknown")

def is_outcome_error(state: "AgentState") -> bool:
    """Check if the agent outcome is an error."""
    return get_outcome_status(state) == "error"

def get_outcome_result(state: "AgentState") -> str:
    """Get the result from the agent outcome."""
    return get_agent_outcome(state).get("result", "")

def get_outcome_agent(state: "AgentState") -> str:
    """Get the agent name from the agent outcome."""
    return get_agent_outcome(state).get("agent", "")

# ============================================================================
# Error & Repair Tracking
# ============================================================================

def get_error_hashes(state: "AgentState") -> list:
    """Get the list of error hashes seen so far."""
    return state.get("error_hashes", [])

def add_error_hash(state: "AgentState", error_hash: str) -> None:
    """Add an error hash to the tracking list."""
    error_hashes = get_error_hashes(state)
    if error_hash not in error_hashes:
        error_hashes.append(error_hash)
    state["error_hashes"] = error_hashes

def was_local_repair_attempted(state: "AgentState") -> bool:
    """Check if a local repair was attempted."""
    return state.get("local_repair_attempted", False)

def mark_repair_attempted(state: "AgentState") -> None:
    """Mark that a local repair was attempted."""
    state["local_repair_attempted"] = True

# ============================================================================
# Action & Routing Control
# ============================================================================

def get_next_action(state: "AgentState") -> str:
    """Get the next planned action."""
    return state.get("next_action", "")

def set_next_action(state: "AgentState", action: str) -> None:
    """Set the next planned action."""
    state["next_action"] = action

def get_selected_agent(state: "AgentState", default: str = "") -> str:
    """Get the selected agent for delegation."""
    return state.get("selected_agent", default)

def set_selected_agent(state: "AgentState", agent: str) -> None:
    """Set the selected agent for delegation."""
    state["selected_agent"] = agent

# ============================================================================
# Tool Call Management
# ============================================================================

def get_tool_calls(state: "AgentState") -> list:
    """Get the list of tool calls."""
    return state.get("tool_calls", [])

def set_tool_calls(state: "AgentState", tool_calls: list) -> None:
    """Set the list of tool calls."""
    state["tool_calls"] = tool_calls

def add_tool_call(state: "AgentState", tool_call: dict) -> None:
    """Add a single tool call to the list."""
    tool_calls = get_tool_calls(state)
    tool_calls.append(tool_call)
    set_tool_calls(state, tool_calls)

def has_tool_call_of_type(state: "AgentState", tool_type: str) -> bool:
    """Check if any tool call matches the given type."""
    return any(call.get("tool") == tool_type for call in get_tool_calls(state))

def get_tool_calls_by_type(state: "AgentState", tool_type: str) -> list:
    """Get all tool calls of a specific type."""
    return [call for call in get_tool_calls(state) if call.get("tool") == tool_type]

# ============================================================================
# Response & Final Output
# ============================================================================

def get_final_response(state: "AgentState") -> Optional[str]:
    """Get the final response to return to the user."""
    return state.get("final_response")

def set_final_response(state: "AgentState", response: Optional[str]) -> None:
    """Set the final response."""
    state["final_response"] = response
