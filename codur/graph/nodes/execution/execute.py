"""Execution node that runs agents with the AgentExecutor."""

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.graph.nodes.types import ExecuteNodeResult
from .agent_executor import AgentExecutor


def execute_node(state: AgentState, config: CodurConfig) -> ExecuteNodeResult:
    """Execution node: Actually run the delegated task.

    Args:
        state: Current agent state with agent_outcome and messages
        config: Codur configuration for agent setup

    Returns:
        Dictionary with agent_outcome containing execution result
    """
    return AgentExecutor(state, config).execute()
