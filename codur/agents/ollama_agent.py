"""
Ollama agent wrapper with async support
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
from rich.console import Console

# Add the mcp-servers directory to path to import ollama_client
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "mcp-servers" / "ollama"))

from ollama_client import OllamaClient
from codur.config import CodurConfig
from codur.agents.base import BaseAgent
from codur.agents import AgentRegistry

console = Console()


class OllamaAgent(BaseAgent):
    """
    Agent that delegates to local Ollama for code generation.

    Supports both sync and async execution, streaming, and various models.
    """

    def __init__(self, config: CodurConfig, override_config: Optional[dict] = None):
        """
        Initialize Ollama agent.

        Args:
            config: Codur configuration
        """
        super().__init__(config, override_config)

        # Get Ollama-specific config
        agent_config = self._get_agent_config("ollama")

        # Determine base URL
        ollama_mcp = config.mcp_servers.get("ollama", {})
        base_url = agent_config.get("base_url", "http://localhost:11434")

        if hasattr(ollama_mcp, "env"):
            base_url = ollama_mcp.env.get("OLLAMA_HOST", base_url)
        ollama_provider = config.providers.get("ollama")
        if ollama_provider and ollama_provider.base_url:
            base_url = ollama_provider.base_url

        # Determine model
        self.agent_config = agent_config
        model = agent_config.get("model", "ministral-3:14b")

        console.print(f"Initializing Ollama agent with model={model}, base_url={base_url}")

        self.client = OllamaClient(
            base_url=base_url,
            model=model,
            temperature=config.llm_temperature,
        )

        self.model = model
        self.base_url = base_url

    def execute(self, task: str, stream: bool = False) -> str:
        """
        Execute a task using Ollama (synchronous).

        Args:
            task: The coding task to perform
            stream: Whether to stream the response

        Returns:
            The generated code or response

        Raises:
            Exception: If Ollama execution fails
        """
        try:
            self._log_execution_start(task)

            prompt = task
            system_prompt = self.agent_config.get("system_prompt")
            if system_prompt:
                prompt = f"{system_prompt}\n\nTask: {task}"
            result = self.client.generate(prompt, stream=stream)

            self._log_execution_complete(result)
            return result

        except Exception as e:
            self._handle_execution_error(e)

    async def aexecute(self, task: str) -> str:
        """
        Execute a task using Ollama (asynchronous).

        Args:
            task: The coding task to perform

        Returns:
            The generated code or response

        Raises:
            Exception: If Ollama execution fails
        """
        try:
            self._log_execution_start(task, is_async=True)

            # Run synchronous client in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.generate(task, stream=False)
            )

            self._log_execution_complete(result, is_async=True)
            return result

        except Exception as e:
            self._handle_execution_error(e)

    async def astream(self, task: str) -> AsyncGenerator[str, None]:
        """
        Execute a task using Ollama with streaming (asynchronous).

        Args:
            task: The coding task to perform

        Yields:
            Chunks of generated text

        Raises:
            Exception: If Ollama execution fails
        """
        try:
            console.print(f"[Async Stream] Executing task with Ollama: {task[:100]}...")

            # Run synchronous streaming in thread pool
            loop = asyncio.get_event_loop()

            def _stream():
                for chunk in self.client.stream_generate(task):
                    yield chunk

            # Convert sync generator to async
            for chunk in _stream():
                yield chunk
                # Allow other tasks to run
                await asyncio.sleep(0)

        except Exception as e:
            self._handle_execution_error(e, "streaming")

    def chat(self, messages: list[dict]) -> str:
        """
        Chat with Ollama using conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Assistant's response

        Raises:
            Exception: If chat fails
        """
        try:
            console.print(f"Chat with Ollama: {len(messages)} messages")

            result = self.client.chat(messages, stream=False)

            console.print(f"Ollama chat response: {len(result)} characters")
            return result

        except Exception as e:
            self._handle_execution_error(e, "chat")

    async def achat(self, messages: list[dict]) -> str:
        """
        Chat with Ollama using conversation history (asynchronous).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Assistant's response

        Raises:
            Exception: If chat fails
        """
        try:
            console.print(f"[Async] Chat with Ollama: {len(messages)} messages")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.chat(messages, stream=False)
            )

            console.print(f"[Async] Ollama chat response: {len(result)} characters")
            return result

        except Exception as e:
            self._handle_execution_error(e, "chat")

    def list_models(self) -> list[dict]:
        """
        List available Ollama models.

        Returns:
            List of model information dicts

        Raises:
            Exception: If listing fails
        """
        try:
            return self.client.list_models()
        except Exception as e:
            self._handle_execution_error(e, "list_models")

    def switch_model(self, model: str):
        """
        Switch to a different Ollama model.

        Args:
            model: Model name to switch to
        """
        console.print(f"Switching Ollama model from {self.model} to {model}")
        self.client.switch_model(model)
        self.model = model

    def __repr__(self) -> str:
        return f"OllamaAgent(model={self.model}, base_url={self.base_url})"

    @property
    def name(self) -> str:
        """Return the agent's name.

        Returns:
            str: "ollama"
        """
        return "ollama"

    @classmethod
    def get_description(cls) -> str:
        """Return a description of the agent's capabilities.

        Returns:
            str: Agent description
        """
        return "Local LLM for code generation (FREE). Best for simple code generation and explanations."


# Register the agent
AgentRegistry.register("ollama", OllamaAgent)
