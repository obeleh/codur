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


# Task-specific temperature defaults
# These provide reasonable defaults for different task types
TASK_TEMPERATURES = {
    "planning": 0.2,          # Deterministic - we want consistent planning decisions
    "explaining": 0.7,        # Balanced - clear explanations with some creativity
    "coding": None,           # Use config default - varied approach for different coding tasks
    "execution": None,        # Use config default - depends on the execution type
    "tool_detection": 0.1,    # Very deterministic - clear tool identification
}


def get_temperature_for_task(task_type: str, config: CodurConfig) -> float:
    """Get the appropriate temperature for a task type.

    Returns task-specific temperature if configured, otherwise falls back to config default.

    Args:
        task_type: Type of task (e.g., "planning", "explaining", "coding")
        config: Codur configuration

    Returns:
        float: Temperature value for the task
    """
    task_temperature = TASK_TEMPERATURES.get(task_type)
    if task_temperature is not None:
        return task_temperature
    return config.llm.default_temperature


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


def create_llm(config: CodurConfig, temperature: float | None = None) -> BaseChatModel:
    """Create an LLM instance using the default profile.

    Args:
        config: Codur configuration
        temperature: Optional temperature override

    Returns:
        BaseChatModel: Configured LLM instance

    Raises:
        ValueError: If no default profile is configured
    """
    if not config.llm.default_profile:
        raise ValueError("No default LLM profile configured (llm.default_profile)")
    return create_llm_profile(config, config.llm.default_profile, temperature=temperature)


def create_llm_profile(
    config: CodurConfig,
    profile_name: str,
    json_mode: bool = False,
    temperature: float | None = None,
    tool_schemas: list[dict] | None = None,
) -> BaseChatModel:
    """Create an LLM instance for a specific profile using the provider registry.

    Args:
        config: Codur configuration
        profile_name: Name of the profile to use
        json_mode: If True, force JSON output format (mutually exclusive with tool_schemas)
        temperature: Optional temperature override (overrides profile and default)
        tool_schemas: Tool schemas to bind (mutually exclusive with json_mode)

    Returns:
        BaseChatModel: Configured LLM instance

    Raises:
        ValueError: If profile doesn't exist, provider is unsupported, or both json_mode and tool_schemas are provided
    """
    # CRITICAL: json_mode and tool_schemas are mutually exclusive
    if json_mode and tool_schemas:
        raise ValueError(
            "json_mode and tool_schemas are mutually exclusive. "
            "json_mode sets tool_choice=none, which conflicts with tool calling."
        )

    profile = config.llm.profiles.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown LLM profile: {profile_name}")

    provider = profile.provider.lower()

    # Resolve temperature: override > profile > default
    if temperature is None:
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

    # Create base LLM (with json_mode if requested and no tools)
    # Try to pass json_mode if provider supports it
    try:
        llm = provider_class.create(config, model, temperature, api_key, json_mode=json_mode)
    except TypeError:
        # Provider doesn't support json_mode, fallback to basic call
        llm = provider_class.create(config, model, temperature, api_key)

    # Bind tools if provided and supported
    if tool_schemas:
        if not provider_class.supports_native_tools():
            raise ValueError(
                f"Provider '{provider}' does not support native tool calling. "
                f"Use json_mode with prompt injection instead."
            )
        llm = provider_class.bind_tools_to_llm(llm, tool_schemas)

    return llm


def create_llm_with_tools(
    config: CodurConfig,
    profile_name: str,
    tool_schemas: list[dict],
    temperature: float | None = None,
) -> BaseChatModel:
    """Create LLM with tools bound (convenience wrapper).

    Args:
        config: Codur configuration
        profile_name: Profile name from config
        tool_schemas: Tool schemas to bind
        temperature: Optional temperature override

    Returns:
        LLM with tools bound

    Example:
        from codur.tools.schema_generator import get_function_schemas

        schemas = get_function_schemas()
        llm = create_llm_with_tools(config, "groq-qwen3-32b", schemas)
        response = llm.invoke([HumanMessage(content="Read main.py")])
        # response.tool_calls will contain structured tool calls
    """
    return create_llm_profile(
        config,
        profile_name,
        json_mode=False,  # Explicitly disabled
        tool_schemas=tool_schemas,
        temperature=temperature,
    )
