"""
LangChain-compatible wrapper for Ollama
"""

from typing import List, Any, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

from codur.agents.ollama_agent import OllamaAgent
from codur.config import CodurConfig


class ChatOllama(BaseChatModel):
    """
    LangChain-compatible chat model wrapper for Ollama.

    This allows using OllamaAgent with LangChain's standard interfaces.
    """

    config: CodurConfig
    agent: Optional[OllamaAgent] = Field(default=None, exclude=True)
    json_mode: bool = Field(default=False)

    def __init__(self, config: CodurConfig, json_mode: bool = False, **kwargs):
        """Initialize with Codur config."""
        super().__init__(config=config, json_mode=json_mode, **kwargs)
        self.agent = OllamaAgent(config)

    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "ollama"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response from messages.

        Args:
            messages: List of messages to send to the model
            stop: Stop sequences (not implemented)
            **kwargs: Additional arguments

        Returns:
            ChatResult with the generated message
        """
        # Convert LangChain messages to Ollama chat format
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"

            ollama_messages.append({
                "role": role,
                "content": msg.content
            })

        # Get response from Ollama with JSON mode if enabled
        format_param = "json" if self.json_mode else None
        response_text = self.agent.client.chat(ollama_messages, format=format_param)

        # Wrap in LangChain format
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of generate."""
        # Convert messages
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"

            ollama_messages.append({
                "role": role,
                "content": msg.content
            })

        # Get async response from Ollama with JSON mode if enabled
        # Note: We need to pass format parameter through the client
        import asyncio
        loop = asyncio.get_event_loop()
        format_param = "json" if self.json_mode else None
        response_text = await loop.run_in_executor(
            None,
            lambda: self.agent.client.chat(ollama_messages, format=format_param)
        )

        # Wrap in LangChain format
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])
