from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage


if TYPE_CHECKING:
    # Avoid circular imports for type checking
    # We don't want to import anything from codur.graph.state or codur.graph.nodes.types at runtime

    from codur.graph.state import AgentState
    from codur.graph.nodes.types import ExecuteNodeResult

def add_message(state: "AgentState", message: BaseMessage) -> None:
    messages = list(state.get("messages", []))
    state["messages"] = messages + [message]

def add_llm_call(state: "AgentState", execute_result: "ExecuteNodeResult") -> None:
    llm_calls = state.get("llm_calls", 0)
    llm_calls += execute_result.get("llm_calls", 0)
    state["llm_calls"] = llm_calls

def get_outcome(execute_result: "ExecuteNodeResult") -> str:
    outcome = execute_result.get("agent_outcome", {})
    if outcome.get("status") == "error":
        raise Exception(f"Agent execution failed: {outcome.get('result')}")
    return outcome.get("result", "")