"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Type
from langchain_core.language_models.chat_models import BaseChatModel
from codur.config import CodurConfig


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Each provider must implement methods to create LLM instances
    and resolve API keys from configuration.
    """

    @staticmethod
    @abstractmethod
    def create(
        config: CodurConfig,
        model: str,
        temperature: float,
        api_key: Optional[str] = None
    ) -> BaseChatModel:
        """Create and return a configured LLM instance.

        Args:
            config: Codur configuration object
            model: Model name/identifier
            temperature: Temperature for generation
            api_key: Optional API key (if None, will be resolved from env)

        Returns:
            BaseChatModel: Configured LangChain chat model instance

        Raises:
            ImportError: If required provider package is not installed
            ValueError: If configuration is invalid
        """
        pass

    @staticmethod
    @abstractmethod
    def get_api_key_env_name(config: CodurConfig) -> str:
        """Return the environment variable name for this provider's API key.

        Args:
            config: Codur configuration object

        Returns:
            str: Environment variable name (e.g., "ANTHROPIC_API_KEY")
        """
        pass

    @staticmethod
    @abstractmethod
    def provider_name() -> str:
        """Return the provider name.

        Returns:
            str: Provider name (e.g., "anthropic", "openai")
        """
        pass

    @staticmethod
    def _resolve_api_key_env(config: CodurConfig, provider_name: str, default_env: str) -> str:
        """Resolve API key environment variable name for a provider.

        This helper method extracts the API key environment variable name from
        configuration, falling back to a default if not specified.

        Args:
            config: Codur configuration object
            provider_name: Provider name to look up (e.g., "groq", "openai")
            default_env: Default environment variable name to use if not in config

        Returns:
            str: Environment variable name for the provider's API key
        """
        provider_cfg = config.providers.get(provider_name)
        if provider_cfg and provider_cfg.api_key_env:
            return provider_cfg.api_key_env
        return default_env


class ProviderRegistry:
    """Registry for LLM providers.

    Allows registration and retrieval of provider implementations
    without using if/elif chains.
    """

    _providers: Dict[str, Type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, provider_name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a provider implementation.

        Args:
            provider_name: Name to register under (e.g., "anthropic")
            provider_class: Provider class to register
        """
        cls._providers[provider_name.lower()] = provider_class

    @classmethod
    def get(cls, provider_name: str) -> Optional[Type[BaseLLMProvider]]:
        """Get a registered provider by name.

        Args:
            provider_name: Provider name to look up

        Returns:
            Optional[Type[BaseLLMProvider]]: Provider class if found, None otherwise
        """
        return cls._providers.get(provider_name.lower())

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            list[str]: List of registered provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers. Useful for testing."""
        cls._providers.clear()
