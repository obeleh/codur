"""
Groq LLM provider implementation.
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from codur.config import CodurConfig
from codur.providers.base import BaseLLMProvider, ProviderRegistry
from codur.providers.utils import lazy_import


class GroqProvider(BaseLLMProvider):
    """Groq provider implementation."""

    @staticmethod
    def create(
        config: CodurConfig,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        json_mode: bool = False
    ) -> BaseChatModel:
        """Create Groq chat model.

        Args:
            config: Codur configuration
            model: Model name (e.g., "llama-3.3-70b-versatile")
            temperature: Generation temperature
            api_key: Optional API key (resolved from env if not provided)
            json_mode: If True, force JSON output format

        Returns:
            BaseChatModel: Configured ChatGroq instance

        Raises:
            RuntimeError: If langchain-groq is not installed
        """
        langchain_groq = lazy_import(
            "langchain_groq",
            "Groq support requires langchain-groq. Install with `pip install langchain-groq`."
        )
        ChatGroq = langchain_groq.ChatGroq

        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key

        # Enable JSON mode if requested
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        return ChatGroq(**kwargs)

    @staticmethod
    def get_api_key_env_name(config: CodurConfig) -> str:
        """Return API key environment variable name.

        Args:
            config: Codur configuration

        Returns:
            str: Environment variable name for Groq API key
        """
        return BaseLLMProvider._resolve_api_key_env(config, "groq", "GROQ_API_KEY")

    @staticmethod
    def provider_name() -> str:
        """Return provider name.

        Returns:
            str: "groq"
        """
        return "groq"

    @staticmethod
    def supports_native_tools() -> bool:
        """Groq supports native tool calling (via OpenAI-compatible API)."""
        return True

    @staticmethod
    def bind_tools_to_llm(llm: BaseChatModel, tool_schemas: list[dict]) -> BaseChatModel:
        """Bind tools for Groq."""
        # Groq uses OpenAI-compatible API
        return llm.bind_tools(tool_schemas)


# Register the provider
ProviderRegistry.register("groq", GroqProvider)
