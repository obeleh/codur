"""
Anthropic (Claude) LLM provider implementation.
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from codur.config import CodurConfig
from codur.providers.base import BaseLLMProvider, ProviderRegistry


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation."""

    @staticmethod
    def create(
        config: CodurConfig,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        json_mode: bool = False
    ) -> BaseChatModel:
        """Create Anthropic chat model.

        Args:
            config: Codur configuration
            model: Model name (e.g., "claude-sonnet-4.5")
            temperature: Generation temperature
            api_key: Optional API key (resolved from env if not provided)
            json_mode: Ignored for Anthropic (no native JSON mode)

        Returns:
            BaseChatModel: Configured ChatAnthropic instance
        """
        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        # Note: Anthropic doesn't have a native JSON mode parameter
        # JSON output is achieved through prompting
        return ChatAnthropic(**kwargs)

    @staticmethod
    def get_api_key_env_name(config: CodurConfig) -> str:
        """Return API key environment variable name.

        Args:
            config: Codur configuration

        Returns:
            str: Environment variable name for Anthropic API key
        """
        provider_cfg = config.providers.get("anthropic")
        if provider_cfg and provider_cfg.api_key_env:
            return provider_cfg.api_key_env
        return "ANTHROPIC_API_KEY"

    @staticmethod
    def provider_name() -> str:
        """Return provider name.

        Returns:
            str: "anthropic"
        """
        return "anthropic"


# Register the provider
ProviderRegistry.register("anthropic", AnthropicProvider)
