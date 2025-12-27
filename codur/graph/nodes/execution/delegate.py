"""Delegation node for routing to agents."""

from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.graph.nodes.types import DelegateNodeResult
from codur.utils.config_helpers import get_default_agent
from codur.utils.validation import require_config
from codur.graph.state_operations import is_verbose, get_selected_agent

console = Console()


def delegate_node(state: AgentState, config: CodurConfig) -> DelegateNodeResult:
    """Delegation node: Route to the appropriate agent.

    Args:
        state: Current agent state containing selected_agent
        config: Codur configuration

    Returns:
        Dictionary with agent_outcome containing agent name and status
    """
    if is_verbose(state):
        console.print("[bold cyan]Delegating task...[/bold cyan]")

    # Use the agent selected by the plan_node, fallback to configured default
    default_agent = get_default_agent(config)
    require_config(
        default_agent,
        "agents.preferences.default_agent",
        "agents.preferences.default_agent must be configured",
    )
    selected_agent = get_selected_agent(state, default_agent)

    return {
        "agent_outcome": {
            "agent": selected_agent,
            "status": "delegated"
        }
    }
