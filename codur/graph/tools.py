"""Tool execution node."""

from langchain_core.messages import SystemMessage

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.node_types import ToolNodeResult
from codur.graph.tool_executor import execute_tool_calls
from codur.graph.state_operations import get_llm_calls, get_tool_calls


def tool_node(state: AgentState, config: CodurConfig) -> ToolNodeResult:
    """Execute local filesystem tools requested by the planner.

    Args:
        state: Current agent state containing tool_calls list
        config: Codur configuration for MCP tool access

    Returns:
        Dictionary with agent_outcome and messages containing tool results
    """
    if "config" not in state:
        raise ValueError("AgentState must include config")

    tool_calls = get_tool_calls(state) or []
    execution = execute_tool_calls(tool_calls, state, config, augment=True, summary_mode="full")
    summary = execution.summary

    return {
        "agent_outcome": {
            "agent": "tools",
            "result": summary,
            "status": "success" if not execution.errors else "error",
        },
        "messages": [SystemMessage(content=f"Tool results:\n{summary}")],
        "llm_calls": get_llm_calls(state),
    }
