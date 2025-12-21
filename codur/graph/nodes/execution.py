"""Delegation, execution, and review nodes."""

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.llm import create_llm_profile, create_llm
from codur.agents import AgentRegistry
from codur.graph.nodes.types import DelegateNodeResult, ExecuteNodeResult, ReviewNodeResult
from codur.graph.nodes.utils import (
    _resolve_agent_profile,
    _resolve_agent_reference,
    _normalize_messages,
)

# Import agents to ensure they are registered
import codur.agents.ollama_agent  # noqa: F401
import codur.agents.codex_agent  # noqa: F401
import codur.agents.claude_code_agent  # noqa: F401

console = Console()


def delegate_node(state: AgentState, config: CodurConfig) -> DelegateNodeResult:
    """Delegation node: Route to the appropriate agent.

    Args:
        state: Current agent state containing selected_agent
        config: Codur configuration

    Returns:
        Dictionary with agent_outcome containing agent name and status
    """
    if state.get("verbose"):
        console.print("[bold cyan]Delegating task...[/bold cyan]")

    # Use the agent selected by the plan_node, fallback to configured default
    default_agent = config.agents.preferences.default_agent or "agent:ollama"
    selected_agent = state.get("selected_agent", default_agent)

    return {
        "agent_outcome": {
            "agent": selected_agent,
            "status": "delegated"
        }
    }


def execute_node(state: AgentState, config: CodurConfig) -> ExecuteNodeResult:
    """Execution node: Actually run the delegated task.

    Args:
        state: Current agent state with agent_outcome and messages
        config: Codur configuration for agent setup

    Returns:
        Dictionary with agent_outcome containing execution result
    """
    default_agent = config.agents.preferences.default_agent or "agent:ollama"
    agent_name = state["agent_outcome"].get("agent", default_agent)
    agent_name, profile_override = _resolve_agent_profile(config, agent_name)
    resolved_agent = _resolve_agent_reference(agent_name)

    if state.get("verbose"):
        console.print(f"[bold green]Executing with {agent_name}...[/bold green]")

    messages = _normalize_messages(state.get("messages"))
    last_message = messages[-1] if messages else None

    if not last_message:
        return {"agent_outcome": {"agent": agent_name, "result": "No task provided", "status": "error"}}

    task = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Route to appropriate agent using registry
    try:
        # Handle LLM profiles directly (llm:profile_name)
        if agent_name.startswith("llm:"):
            profile_name = agent_name.split(":", 1)[1]
            llm = create_llm_profile(config, profile_name)
            response = llm.invoke([HumanMessage(content=task)])
            result = response.content
        else:
            # Check if this agent is configured as type "llm" in agents.configs
            agent_config = config.agents.configs.get(resolved_agent)
            if agent_config and hasattr(agent_config, 'type') and agent_config.type == "llm":
                # This is an LLM agent, get the model from config
                model = agent_config.config.get("model")
                # Create LLM instance using the agent's config
                # First try to find an LLM profile that matches this model
                matching_profile = None
                for profile_name, profile in config.llm.profiles.items():
                    if profile.model == model:
                        matching_profile = profile_name
                        break

                if matching_profile:
                    llm = create_llm_profile(config, matching_profile)
                else:
                    # Fallback: use default profile
                    llm = create_llm(config)

                response = llm.invoke([HumanMessage(content=task)])
                result = response.content
            else:
                # Get agent from registry
                agent_class = AgentRegistry.get(resolved_agent)
                if not agent_class:
                    available = ", ".join(AgentRegistry.list_agents())
                    return {
                        "agent_outcome": {
                            "agent": agent_name,
                            "result": f"Unknown agent: {resolved_agent}. Available agents: {available}",
                            "status": "error"
                        }
                    }

                # Create agent instance and execute
                agent = agent_class(config, override_config=profile_override)
                result = agent.execute(task)

        return {
            "agent_outcome": {
                "agent": agent_name,
                "result": result,
                "status": "success"
            }
        }

    except Exception as e:
        console.print(f"[red]Error executing {agent_name}: {str(e)}[/red]")
        return {
            "agent_outcome": {
                "agent": agent_name,
                "result": str(e),
                "status": "error"
            }
        }


def review_node(state: AgentState, llm: BaseChatModel, config: CodurConfig) -> ReviewNodeResult:
    """Review node: Check if the result is satisfactory.

    Args:
        state: Current agent state with agent_outcome
        llm: Language model (unused but kept for compatibility)
        config: Codur configuration (unused but kept for compatibility)

    Returns:
        Dictionary with final_response and next_action set to "end"

    Note:
        Currently accepts all results without quality checks.
        TODO: Add quality checks and iteration logic
    """
    if state.get("verbose"):
        console.print("[bold magenta]Reviewing result...[/bold magenta]")

    outcome = state.get("agent_outcome", {})
    result = outcome.get("result", "")

    return {
        "final_response": result,
        "next_action": "end",
    }
