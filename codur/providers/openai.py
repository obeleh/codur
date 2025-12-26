"""
OpenAI LLM provider implementation.
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from codur.config import CodurConfig
from codur.providers.base import BaseLLMProvider, ProviderRegistry
from codur.providers.utils import lazy_import


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    @staticmethod
    def create(
        config: CodurConfig,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        json_mode: bool = False
    ) -> BaseChatModel:
        """Create OpenAI chat model.

        Args:
            config: Codur configuration
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Generation temperature
            api_key: Optional API key (resolved from env if not provided)
            json_mode: If True, force JSON output format

        Returns:
            BaseChatModel: Configured ChatOpenAI instance

        Raises:
            RuntimeError: If langchain-openai is not installed
        """
        langchain_openai = lazy_import(
            "langchain_openai",
            "OpenAI support requires langchain-openai. Install with `pip install langchain-openai`."
        )
        ChatOpenAI = langchain_openai.ChatOpenAI

        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key

        # Enable JSON mode if requested
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        return ChatOpenAI(**kwargs)

    @staticmethod
    def get_api_key_env_name(config: CodurConfig) -> str:
        """Return API key environment variable name.

        Args:
            config: Codur configuration

        Returns:
            str: Environment variable name for OpenAI API key
        """
        return BaseLLMProvider._resolve_api_key_env(config, "openai", "OPENAI_API_KEY")

    @staticmethod
    def provider_name() -> str:
        """Return provider name.

        Returns:
            str: "openai"
        """
        return "openai"


# Register the provider
ProviderRegistry.register("openai", OpenAIProvider)
