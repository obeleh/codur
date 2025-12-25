"""
Codex agent wrapper with async support
"""

import subprocess
import logging
from typing import Optional

from codur.config import CodurConfig
from codur.agents.cli_agent_base import BaseCLIAgent
from codur.agents import AgentRegistry

logger = logging.getLogger(__name__)


class CodexAgent(BaseCLIAgent):
    """
    Agent that delegates to OpenAI Codex for complex code tasks.

    Supports async execution and various Codex models.
    """

    def __init__(self, config: CodurConfig, override_config: Optional[dict] = None):
        """
        Initialize Codex agent.

        Args:
            config: Codur configuration
        """
        super().__init__(config, override_config)

        # Get Codex-specific config
        codex_config = config.agents.configs.get("codex", {})
        agent_config = codex_config.config if hasattr(codex_config, "config") else {}
        if override_config:
            agent_config = {**agent_config, **override_config}

        self.command = agent_config.get("command", "codex")
        self.model = agent_config.get("model", "gpt-5-codex")
        self.reasoning_effort = agent_config.get("reasoning_effort", "medium")
        self.default_timeout = config.agent_execution.default_cli_timeout

        logger.info(f"Initializing Codex agent with model={self.model}, reasoning={self.reasoning_effort}")

    def _build_command(self, task: str, sandbox: str = "workspace-write") -> list[str]:
        return [
            self.command,
            "exec",
            "--skip-git-repo-check",
            "-m", self.model,
            "--config", f"model_reasoning_effort={self.reasoning_effort}",
            "--sandbox", sandbox,
            "--full-auto",
            task,
        ]

    def execute(
        self,
        task: str,
        sandbox: str = "workspace-write",
        timeout: Optional[int] = None,
    ) -> str:
        """
        Execute a task using Codex CLI (synchronous).

        Args:
            task: The coding task to perform
            sandbox: Sandbox mode (read-only, workspace-write, danger-full-access)
            timeout: Timeout in seconds

        Returns:
            The result from Codex

        Raises:
            Exception: If Codex execution fails
        """
        logger.info(f"Executing task with Codex: {task[:100]}...")
        cmd = self._build_command(task, sandbox=sandbox)
        output = self._execute_cli(cmd, timeout=timeout, suppress_stderr=True)
        logger.info(f"Codex generated {len(output)} characters")
        return output

    async def aexecute(
        self,
        task: str,
        sandbox: str = "workspace-write",
        timeout: Optional[int] = None,
    ) -> str:
        """
        Execute a task using Codex CLI (asynchronous).

        Args:
            task: The coding task to perform
            sandbox: Sandbox mode
            timeout: Timeout in seconds

        Returns:
            The result from Codex

        Raises:
            Exception: If Codex execution fails
        """
        logger.info(f"[Async] Executing task with Codex: {task[:100]}...")
        cmd = self._build_command(task, sandbox=sandbox)
        output = await self._aexecute_cli(cmd, timeout=timeout, suppress_stderr=True)
        logger.info(f"[Async] Codex generated {len(output)} characters")
        return output

    async def astream(self, task: str, sandbox: str = "workspace-write"):
        """
        Execute with streaming output (not yet implemented for Codex).

        Args:
            task: The coding task
            sandbox: Sandbox mode

        Raises:
            NotImplementedError: Streaming not yet supported
        """
        # TODO: Implement streaming for Codex
        raise NotImplementedError("Codex streaming not yet implemented")

    def resume_last(self, additional_prompt: Optional[str] = None) -> str:
        """
        Resume the last Codex session.

        Args:
            additional_prompt: Additional guidance for the resumed session

        Returns:
            The result from Codex

        Raises:
            Exception: If resume fails
        """
        try:
            logger.info("Resuming last Codex session")

            cmd = [self.command, "exec", "--skip-git-repo-check", "resume", "--last"]

            if additional_prompt:
                # Pipe the prompt via stdin
                result = subprocess.run(
                    cmd,
                    input=additional_prompt,
                    capture_output=True,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )

            if result.returncode != 0:
                raise Exception(f"Codex resume failed with code {result.returncode}")

            return result.stdout.strip()

        except Exception as e:
            logger.error(f"Codex resume failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to resume Codex: {str(e)}") from e

    def __repr__(self) -> str:
        return f"CodexAgent(model={self.model}, reasoning={self.reasoning_effort})"

    @property
    def name(self) -> str:
        """Return the agent's name.

        Returns:
            str: "codex"
        """
        return "codex"

    @classmethod
    def get_description(cls) -> str:
        """Return a description of the agent's capabilities.

        Returns:
            str: Agent description
        """
        return "OpenAI Codex for complex code tasks. Best for code refactoring, bug fixes, and optimization."


# Register the agent
AgentRegistry.register("codex", CodexAgent)
