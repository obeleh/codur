"""
Abstract base class for agents in the Codur orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from codur.config import CodurConfig


class BaseAgent(ABC):
    """Abstract base class for all agents in Codur.

    All agents must implement the execute method for synchronous execution
    and aexecute for asynchronous execution. This ensures a consistent
    interface across all agent implementations.
    """

    def __init__(self, config: CodurConfig, override_config: Optional[dict] = None):
        """Initialize the agent with configuration.

        Args:
            config: Main Codur configuration
            override_config: Optional configuration overrides
        """
        self.config = config
        self.override_config = override_config or {}

    @abstractmethod
    def execute(self, task: str, **kwargs: Any) -> str:
        """Execute a task synchronously.

        Args:
            task: The task to execute
            **kwargs: Additional agent-specific parameters

        Returns:
            str: The result of the execution

        Raises:
            Exception: If execution fails
        """
        pass

    @abstractmethod
    async def aexecute(self, task: str, **kwargs: Any) -> str:
        """Execute a task asynchronously.

        Args:
            task: The task to execute
            **kwargs: Additional agent-specific parameters

        Returns:
            str: The result of the execution

        Raises:
            Exception: If execution fails
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the agent.

        Returns:
            str: Human-readable representation
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name.

        Returns:
            str: The agent name
        """
        pass
