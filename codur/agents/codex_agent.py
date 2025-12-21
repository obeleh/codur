"""
Codex agent wrapper with async support
"""

import asyncio
import subprocess
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from codur.config import CodurConfig
from codur.agents.base import BaseAgent
from codur.agents import AgentRegistry

logger = logging.getLogger(__name__)


class CodexAgent(BaseAgent):
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
        self.config = config

        # Get Codex-specific config
        codex_config = config.agents.configs.get("codex", {})
        agent_config = codex_config.config if hasattr(codex_config, "config") else {}
        if override_config:
            agent_config = {**agent_config, **override_config}

        self.command = agent_config.get("command", "codex")
        self.model = agent_config.get("model", "gpt-5-codex")
        self.reasoning_effort = agent_config.get("reasoning_effort", "medium")

        logger.info(f"Initializing Codex agent with model={self.model}, reasoning={self.reasoning_effort}")

    def execute(
        self,
        task: str,
        sandbox: str = "workspace-write",
        timeout: Optional[int] = None
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
        try:
            logger.info(f"Executing task with Codex: {task[:100]}...")

            # Build command
            cmd = [
                self.command,
                "exec",
                "--skip-git-repo-check",
                "-m", self.model,
                "--config", f"model_reasoning_effort={self.reasoning_effort}",
                "--sandbox", sandbox,
                "--full-auto",
                task
            ]

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or 600,
                stderr=subprocess.DEVNULL,  # Suppress thinking tokens
            )

            if result.returncode != 0:
                raise Exception(f"Codex exited with code {result.returncode}: {result.stderr}")

            output = result.stdout.strip()
            logger.info(f"Codex generated {len(output)} characters")

            return output

        except subprocess.TimeoutExpired:
            logger.error(f"Codex execution timed out after {timeout}s")
            raise Exception(f"Codex timed out after {timeout} seconds")

        except FileNotFoundError:
            logger.error(f"Codex command not found: {self.command}")
            raise Exception(
                f"Codex not found. Install it or configure the command path. "
                f"Current command: {self.command}"
            )

        except Exception as e:
            logger.error(f"Codex execution failed: {str(e)}", exc_info=True)
            raise Exception(f"Codex agent failed: {str(e)}") from e

    async def aexecute(
        self,
        task: str,
        sandbox: str = "workspace-write",
        timeout: Optional[int] = None
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
        try:
            logger.info(f"[Async] Executing task with Codex: {task[:100]}...")

            # Build command
            cmd = [
                self.command,
                "exec",
                "--skip-git-repo-check",
                "-m", self.model,
                "--config", f"model_reasoning_effort={self.reasoning_effort}",
                "--sandbox", sandbox,
                "--full-auto",
                task
            ]

            # Create subprocess
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,  # Suppress thinking tokens
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout or 600
                )

                if proc.returncode != 0:
                    raise Exception(f"Codex exited with code {proc.returncode}")

                output = stdout.decode().strip()
                logger.info(f"[Async] Codex generated {len(output)} characters")

                return output

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise Exception(f"Codex timed out after {timeout} seconds")

        except FileNotFoundError:
            logger.error(f"Codex command not found: {self.command}")
            raise Exception(
                f"Codex not found. Install it or configure the command path. "
                f"Current command: {self.command}"
            )

        except Exception as e:
            logger.error(f"[Async] Codex execution failed: {str(e)}", exc_info=True)
            raise Exception(f"Codex agent failed: {str(e)}") from e

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


# Register the agent
AgentRegistry.register("codex", CodexAgent)
