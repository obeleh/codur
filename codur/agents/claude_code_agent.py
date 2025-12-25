"""
Claude Code agent wrapper with async support

Allows Codur to delegate tasks to Claude Code CLI for:
- Code generation and editing
- File operations
- Complex reasoning
- Tool usage (bash, read, write, etc.)
"""

import logging
from typing import Optional

from codur.config import CodurConfig
from codur.agents.cli_agent_base import BaseCLIAgent
from codur.agents import AgentRegistry

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(BaseCLIAgent):
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
        super().__init__(config, override_config)

        # Get Claude Code-specific config
        claude_config = config.agents.configs.get("claude_code", {})
        agent_config = claude_config.config if hasattr(claude_config, "config") else {}
        if override_config:
            agent_config = {**agent_config, **override_config}

        self.command = agent_config.get("command", "claude")
        self.model = agent_config.get("model", "sonnet")  # sonnet, opus, haiku
        self.max_tokens = agent_config.get(
            "max_tokens",
            config.agent_execution.claude_code_max_tokens,
        )
        self.default_timeout = config.agent_execution.default_cli_timeout

        logger.info(f"Initializing Claude Code agent with model={self.model}")

    def _build_command(self, prompt: str, files: Optional[list[str]] = None) -> list[str]:
        cmd = [
            self.command,
            "chat",
            "--model", self.model,
            "--max-tokens", str(self.max_tokens),
            "--message", prompt,
        ]
        if files:
            for file_path in files:
                cmd.extend(["--file", file_path])
        return cmd

    def _not_found_message(self) -> str:
        return (
            "Claude Code not found. Install it from https://claude.com/claude-code. "
            f"Current command: {self.command}"
        )

    def execute(
        self,
        task: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None,
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
        logger.info(f"Executing task with Claude Code: {task[:100]}...")
        prompt = self._build_prompt(task, context)
        cmd = self._build_command(prompt)
        output = self._execute_cli(cmd, timeout=timeout, capture_stderr=True)
        logger.info(f"Claude Code generated {len(output)} characters")
        return output

    async def aexecute(
        self,
        task: str,
        context: Optional[str] = None,
        timeout: Optional[int] = None,
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
        logger.info(f"[Async] Executing task with Claude Code: {task[:100]}...")
        prompt = self._build_prompt(task, context)
        cmd = self._build_command(prompt)
        output = await self._aexecute_cli(cmd, timeout=timeout, capture_stderr=True)
        logger.info(f"[Async] Claude Code generated {len(output)} characters")
        return output

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

            cmd = self._build_command(prompt, files=files)
            return self._execute_cli(cmd, timeout=600, capture_stderr=True)

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

    @classmethod
    def get_description(cls) -> str:
        """Return a description of the agent's capabilities.

        Returns:
            str: Agent description
        """
        return "Claude Code CLI integration. Best for multi-file changes, complex reasoning, and tool usage."


# Register the agent
AgentRegistry.register("claude_code", ClaudeCodeAgent)
