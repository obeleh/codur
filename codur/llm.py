"""
LLM factory for the Codur orchestrator.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from codur.config import CodurConfig
from codur.providers.base import ProviderRegistry

# Import all providers to ensure they are registered
import codur.providers.ollama  # noqa: F401
import codur.providers.anthropic  # noqa: F401
import codur.providers.groq  # noqa: F401
import codur.providers.openai  # noqa: F401


def _resolve_api_key(config: CodurConfig, provider: str, api_key_env: str | None = None) -> str | None:
    """Resolve API key from environment using provider registry.

    Args:
        config: Codur configuration
        provider: Provider name
        api_key_env: Optional explicit environment variable name

    Returns:
        Optional[str]: API key from environment, or None
    """
    if api_key_env:
        return os.getenv(api_key_env)

    provider_class = ProviderRegistry.get(provider.lower())
    if provider_class:
        env_name = provider_class.get_api_key_env_name(config)
        return os.getenv(env_name) if env_name else None

    return None


def create_llm(config: CodurConfig) -> BaseChatModel:
    """Create an LLM instance using the default profile.

    Args:
        config: Codur configuration

    Returns:
        BaseChatModel: Configured LLM instance

    Raises:
        ValueError: If no default profile is configured
    """
    if not config.llm.default_profile:
        raise ValueError("No default LLM profile configured (llm.default_profile)")
    return create_llm_profile(config, config.llm.default_profile)


def create_llm_profile(config: CodurConfig, profile_name: str, json_mode: bool = False) -> BaseChatModel:
    """Create an LLM instance for a specific profile using the provider registry.

    Args:
        config: Codur configuration
        profile_name: Name of the profile to use
        json_mode: If True, force JSON output format (supported by Groq and Ollama)

    Returns:
        BaseChatModel: Configured LLM instance

    Raises:
        ValueError: If profile doesn't exist or provider is unsupported
    """
    profile = config.llm.profiles.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown LLM profile: {profile_name}")

    provider = profile.provider.lower()
    temperature = profile.temperature if profile.temperature is not None else config.llm.default_temperature
    model = profile.model
    api_key = _resolve_api_key(config, provider, profile.api_key_env)

    # Get provider from registry
    provider_class = ProviderRegistry.get(provider)
    if not provider_class:
        available = ", ".join(ProviderRegistry.list_providers())
        raise ValueError(
            f"Unsupported LLM provider for profile {profile_name}: {provider}. "
            f"Available providers: {available}"
        )

    # Use provider to create LLM instance
    # Try to pass json_mode if provider supports it
    try:
        return provider_class.create(config, model, temperature, api_key, json_mode=json_mode)
    except TypeError:
        # Provider doesn't support json_mode, fallback to basic call
        return provider_class.create(config, model, temperature, api_key)
