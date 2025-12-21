"""
Agent implementations and registry for Codur.
"""

from typing import Dict, Type, Optional
from codur.agents.base import BaseAgent


class AgentRegistry:
    """Registry for agent implementations.

    Allows registration and retrieval of agent implementations
    without using if/elif chains.
    """

    _agents: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, agent_name: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent implementation.

        Args:
            agent_name: Name to register under (e.g., "ollama", "codex")
            agent_class: Agent class to register
        """
        cls._agents[agent_name.lower()] = agent_class

    @classmethod
    def get(cls, agent_name: str) -> Optional[Type[BaseAgent]]:
        """Get a registered agent by name.

        Args:
            agent_name: Agent name to look up

        Returns:
            Optional[Type[BaseAgent]]: Agent class if found, None otherwise
        """
        return cls._agents.get(agent_name.lower())

    @classmethod
    def list_agents(cls) -> list[str]:
        """List all registered agent names.

        Returns:
            list[str]: List of registered agent names
        """
        return list(cls._agents.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents. Useful for testing."""
        cls._agents.clear()


__all__ = ["BaseAgent", "AgentRegistry"]
