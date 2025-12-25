"""Agent execution tools."""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.state_operations import add_message, add_llm_call, get_outcome

def _handoff_to_executor(
    agent: str,
    state: AgentState,
    config: CodurConfig
) -> str:
    from codur.graph.nodes.execution import AgentExecutor
    executor = AgentExecutor(state, config, agent_name=agent)
    result = executor.execute()
    add_llm_call(state, result)
    return get_outcome(result)


# TODO: Rename from challenge to task for consistency

def agent_call(
    agent: str,
    challenge: str,
    state: AgentState,
    config: CodurConfig,
) -> str:
    from codur.graph.nodes.execution import AgentExecutor

    """Invoke an agent with a coding challenge and optional file context.

    Used by the planner to directly invoke agents with file contents after reading them.

    Args:
        agent: Agent name or reference (agent:<name> or llm:<profile>)
        challenge: The coding challenge/task description
        state: Current agent state for tracking calls and verbosity
        config: Codur configuration for agent setup

    Returns:
        Agent response string
    """
    add_message(state, HumanMessage(content=challenge))
    return _handoff_to_executor(state=state, agent=agent, config=config)


def retry_in_agent(
    agent: str,
    task: str,
    state: AgentState,
    config: CodurConfig,
    reason: Optional[str] = None,
) -> str:
    """Retry a task using a specific agent name.

    Args:
        agent: Agent name or reference (agent:<name> or llm:<profile>)
        task: Task content to run
        state: Current agent state for tracking calls and verbosity
        config: Codur configuration for agent setup
        reason: Optional reason for the retry

    Returns:
        Agent response string
    """
    if reason:
        system_message = f"Previous attempt failed due to: {reason}. Please try again."
    else:
        system_message = f"Something went wrong We're retrying in agent {agent}"
    add_message(state, SystemMessage(content=system_message))
    add_message(state, HumanMessage(content=task))
    return _handoff_to_executor(state=state, agent=agent, config=config)