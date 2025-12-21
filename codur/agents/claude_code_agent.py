"""
Claude Code agent wrapper with async support

Allows Codur to delegate tasks to Claude Code CLI for:
- Code generation and editing
- File operations
- Complex reasoning
- Tool usage (bash, read, write, etc.)
"""

import asyncio
import subprocess
import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path

from codur.config import CodurConfig
from codur.agents.base import BaseAgent
from codur.agents import AgentRegistry

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(BaseAgent):
    """
    Agent that delegates to Claude Code CLI for complex tasks.

    Claude Code is excellent for:
    - Multi-file code changes
    - Complex refactoring
    - File system operations
    - Using tools (bash, grep, read, write)
    - Reasoning about code architecture
    """

    def __init__(self, config: CodurConfig, override_config: Optional[dict] = None):
        """
        Initialize Claude Code agent.

        Args:
            config: Codur configuration
        """
        self.config = config

        # Get Claude Code-specific config
        claude_config = config.agents.configs.get("claude_code", {})
        agent_config = claude_config.config if hasattr(claude_config, "config") else {}
        if override_config:
            agent_config = {**agent_config, **override_config}

        self.command = agent_config.get("command", "claude")
        self.model = agent_config.get("model", "sonnet")  # sonnet, opus, haiku
        self.max_tokens = agent_config.get("max_tokens", 8000)

        logger.info(f"Initializing Claude Code agent with model={self.model}")

    def execute(
        self,
        task: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Execute a task using Claude Code CLI (synchronous).

        Args:
            task: The task to execute
            context: Additional context or instructions
            timeout: Timeout in seconds

        Returns:
            The result from Claude Code

        Raises:
            Exception: If execution fails
        """
        try:
            logger.info(f"Executing task with Claude Code: {task[:100]}...")

            # Build the full prompt
            prompt = self._build_prompt(task, context)

            # Build command
            cmd = [
                self.command,
                "chat",
                "--model", self.model,
                "--max-tokens", str(self.max_tokens),
                "--message", prompt
            ]

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or 600,
            )

            if result.returncode != 0:
                raise Exception(f"Claude Code exited with code {result.returncode}: {result.stderr}")

            output = result.stdout.strip()
            logger.info(f"Claude Code generated {len(output)} characters")

            return output

        except subprocess.TimeoutExpired:
            logger.error(f"Claude Code execution timed out after {timeout}s")
            raise Exception(f"Claude Code timed out after {timeout} seconds")

        except FileNotFoundError:
            logger.error(f"Claude Code command not found: {self.command}")
            raise Exception(
                f"Claude Code not found. Install it from https://claude.com/claude-code. "
                f"Current command: {self.command}"
            )

        except Exception as e:
            logger.error(f"Claude Code execution failed: {str(e)}", exc_info=True)
            raise Exception(f"Claude Code agent failed: {str(e)}") from e

    async def aexecute(
        self,
        task: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Execute a task using Claude Code CLI (asynchronous).

        Args:
            task: The task to execute
            context: Additional context
            timeout: Timeout in seconds

        Returns:
            The result from Claude Code

        Raises:
            Exception: If execution fails
        """
        try:
            logger.info(f"[Async] Executing task with Claude Code: {task[:100]}...")

            prompt = self._build_prompt(task, context)

            # Build command
            cmd = [
                self.command,
                "chat",
                "--model", self.model,
                "--max-tokens", str(self.max_tokens),
                "--message", prompt
            ]

            # Create subprocess
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout or 600
                )

                if proc.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    raise Exception(f"Claude Code exited with code {proc.returncode}: {error_msg}")

                output = stdout.decode().strip()
                logger.info(f"[Async] Claude Code generated {len(output)} characters")

                return output

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise Exception(f"Claude Code timed out after {timeout} seconds")

        except FileNotFoundError:
            logger.error(f"Claude Code command not found: {self.command}")
            raise Exception(
                f"Claude Code not found. Install it from https://claude.com/claude-code"
            )

        except Exception as e:
            logger.error(f"[Async] Claude Code execution failed: {str(e)}", exc_info=True)
            raise Exception(f"Claude Code agent failed: {str(e)}") from e

    def execute_with_files(
        self,
        task: str,
        files: list[str],
        context: Optional[str] = None
    ) -> str:
        """
        Execute a task with specific files in context.

        Args:
            task: The task to execute
            files: List of file paths to include in context
            context: Additional instructions

        Returns:
            The result from Claude Code

        Raises:
            Exception: If execution fails
        """
        try:
            logger.info(f"Executing with {len(files)} files: {task[:100]}...")

            prompt = self._build_prompt(task, context)

            # Add file context
            if files:
                prompt += "\n\nFiles to consider:\n"
                for file_path in files:
                    prompt += f"- {file_path}\n"

            # Build command
            cmd = [
                self.command,
                "chat",
                "--model", self.model,
                "--max-tokens", str(self.max_tokens),
                "--message", prompt
            ]

            # Add file arguments
            for file_path in files:
                cmd.extend(["--file", file_path])

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise Exception(f"Claude Code exited with code {result.returncode}")

            return result.stdout.strip()

        except Exception as e:
            logger.error(f"Claude Code file execution failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to execute with files: {str(e)}") from e

    def _build_prompt(self, task: str, context: Optional[str] = None) -> str:
        """
        Build the full prompt for Claude Code.

        Args:
            task: The main task
            context: Additional context

        Returns:
            The complete prompt
        """
        prompt = task

        if context:
            prompt = f"{context}\n\n{task}"

        # Add instructions for Claude Code
        prompt += "\n\nPlease provide a complete, working solution."

        return prompt

    def chat(
        self,
        messages: list[dict],
        stream: bool = False
    ) -> str:
        """
        Multi-turn chat with Claude Code.

        Args:
            messages: List of messages with 'role' and 'content'
            stream: Whether to stream (not yet supported)

        Returns:
            Assistant's response

        Raises:
            Exception: If chat fails
        """
        try:
            logger.info(f"Chat with Claude Code: {len(messages)} messages")

            # For now, just use the last user message
            # TODO: Implement proper multi-turn chat with history
            last_user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content")
                    break

            if not last_user_msg:
                raise Exception("No user message found in chat history")

            return self.execute(last_user_msg)

        except Exception as e:
            logger.error(f"Claude Code chat failed: {str(e)}", exc_info=True)
            raise Exception(f"Chat failed: {str(e)}") from e

    def __repr__(self) -> str:
        return f"ClaudeCodeAgent(model={self.model}, command={self.command})"

    @property
    def name(self) -> str:
        """Return the agent's name.

        Returns:
            str: "claude_code"
        """
        return "claude_code"


# Register the agent
AgentRegistry.register("claude_code", ClaudeCodeAgent)
