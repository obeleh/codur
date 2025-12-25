"""
Abstract base class for agents in the Codur orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from rich.console import Console

from codur.config import CodurConfig

console = Console()


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

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Return a description of the agent's capabilities.

        Returns:
            str: Agent description
        """
        pass

    # ========================================================================
    # Shared Helper Methods
    # ========================================================================

    def _get_agent_config(self, agent_name: str) -> dict:
        """Extract and merge agent-specific configuration.

        Args:
            agent_name: Name of the agent in config

        Returns:
            dict: Merged agent configuration
        """
        agent_config_obj = self.config.agents.configs.get(agent_name, {})
        agent_config = agent_config_obj.config if hasattr(agent_config_obj, "config") else {}
        if self.override_config:
            agent_config = {**agent_config, **self.override_config}
        return agent_config

    def _log_execution_start(self, task: str, is_async: bool = False) -> None:
        """Log the start of task execution.

        Args:
            task: Task being executed
            is_async: Whether execution is asynchronous
        """
        prefix = "[Async] " if is_async else ""
        console.print(f"{prefix}Executing task with {self.name}: {task[:100]}...")

    def _log_execution_complete(self, output: str, is_async: bool = False) -> None:
        """Log the completion of task execution.

        Args:
            output: Output from execution
            is_async: Whether execution was asynchronous
        """
        prefix = "[Async] " if is_async else ""
        console.print(f"{prefix}{self.name} generated {len(output)} characters")

    def _handle_execution_error(self, error: Exception, operation: str = "execution") -> Exception:
        """Handle execution errors with standardized formatting.

        Args:
            error: The exception that occurred
            operation: Description of what operation failed

        Returns:
            Exception: A new exception with standardized formatting
        """
        error_msg = f"{self.name} {operation} failed: {str(error)}"
        console.print(f"[red]{error_msg}[/red]")
        raise Exception(error_msg) from error
