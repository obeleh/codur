from typing import TYPE_CHECKING, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from codur.graph.utils import normalize_messages

if TYPE_CHECKING:
    # Avoid circular imports for type checking
    # We don't want to import anything from codur.graph.state or codur.graph.node_types at runtime

    from codur.graph.state import AgentState
    from codur.graph.node_types import ExecuteNodeResult
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

def add_messages(state: "AgentState", new_messages: list[BaseMessage]) -> None:
    """Add multiple messages to the state's message history."""
    messages = list(state.get("messages", []))
    state["messages"] = messages + new_messages

def get_messages(state: "AgentState") -> list[BaseMessage]:
    """Get normalized messages from state."""
    messages = state.get("messages", [])
    return normalize_messages(messages)

def set_messages(state: "AgentState", messages: list[BaseMessage]) -> None:
    """
    Set the state's message history to the given messages.
    Usually not to be used since state mutations do not persist across nodes.
    """
    state["messages"] = messages

def get_first_human_message_from_messages(messages: list[BaseMessage]) -> Optional[BaseMessage]:
    if not messages:
        return None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg
    return None


def get_first_human_message(state: "AgentState") -> Optional[BaseMessage]:
    """Get the first human message from state (original request), if any."""
    messages = get_messages(state)
    return get_first_human_message_from_messages(messages)


def get_last_human_message_from_messages(messages: list[BaseMessage]) -> Optional[BaseMessage]:
    if not messages:
        return None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None

def get_last_human_message(state: "AgentState") -> Optional[BaseMessage]:
    """Get the last human message from state, if any."""
    messages = get_messages(state)
    return get_last_human_message_from_messages(messages)


def get_first_human_message_content(state: "AgentState") -> Optional[str]:
    """Get the content of the first human message (original request), if any."""
    msg = get_first_human_message(state)
    return msg.content if msg else None

def get_last_human_message_content(state: "AgentState") -> Optional[str]:
    """Get the content of the last human message, if any."""
    msg = get_last_human_message(state)
    return msg.content if msg else None

def get_first_human_message_content_from_messages(messages: list[BaseMessage]) -> Optional[str]:
    """Get the content of the first human message from a list of messages, if any."""
    msg = get_first_human_message_from_messages(messages)
    return msg.content if msg else None

def get_last_human_message_content_from_messages(messages: list[BaseMessage]) -> Optional[str]:
    """Get the content of the last human message from a list of messages, if any."""
    msg = get_last_human_message_from_messages(messages)
    return msg.content if msg else None


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

def get_next_step_suggestion(state: "AgentState") -> Optional[str]:
    """Get the next step suggestion from state, if any."""
    return state.get("next_step_suggestion")

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
