"""Shared helpers for graph nodes."""

from typing import Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from codur.config import CodurConfig
from codur.llm import create_llm_profile


def resolve_agent_reference(raw_agent: str) -> str:
    """Extract the actual agent name from agent reference format (e.g., 'agent:ollama' -> 'ollama')."""
    if raw_agent.startswith("agent:"):
        return raw_agent.split(":", 1)[1]
    return raw_agent


def resolve_agent_profile(config: CodurConfig, agent_name: str) -> tuple[str, Optional[dict]]:
    """Resolve agent profile configuration from config.

    Args:
        config: Codur configuration object
        agent_name: Agent name, potentially in 'agent:name' format

    Returns:
        Tuple of (resolved_agent_name, profile_config_dict or None)
    """
    if agent_name.startswith("agent:"):
        profile_name = agent_name.split(":", 1)[1]
        profile = config.agents.profiles.get(profile_name)
        if profile and profile.name:
            return profile.name, profile.config
    return agent_name, None


def normalize_messages(messages: Any) -> list[BaseMessage]:
    """Coerce message-like inputs into LangChain BaseMessage objects.

    Args:
        messages: List of messages in various formats (BaseMessage, dict, or str)

    Returns:
        List of LangChain BaseMessage objects
    """
    normalized: list[BaseMessage] = []
    for message in messages or []:
        if isinstance(message, BaseMessage):
            normalized.append(message)
            continue
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "assistant":
                normalized.append(AIMessage(content=content))
            elif role == "system":
                normalized.append(SystemMessage(content=content))
            else:
                normalized.append(HumanMessage(content=content))
            continue
        normalized.append(HumanMessage(content=str(message)))
    return normalized


def get_first_human_message(messages: list[BaseMessage]) -> Optional[str]:
    """Extract content from first HumanMessage in a list.

    Args:
        messages: List of BaseMessage objects

    Returns:
        Optional[str]: Content of first HumanMessage, or None if not found
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def get_last_human_message(messages: list[BaseMessage]) -> Optional[str]:
    """Extract content from last HumanMessage in a list.

    Args:
        messages: List of BaseMessage objects

    Returns:
        Optional[str]: Content of last HumanMessage, or None if not found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def extract_messages_by_type(messages: list[BaseMessage], message_type: type) -> list[BaseMessage]:
    """Extract all messages of a specific type.

    Args:
        messages: List of BaseMessage objects
        message_type: Type of message to filter for (e.g., HumanMessage, AIMessage)

    Returns:
        List of messages matching the type
    """
    return [msg for msg in messages if isinstance(msg, message_type)]


def resolve_llm_for_model(config: CodurConfig, model: str | None, temperature: float | None = None, json_mode: bool = False):
    """Resolve LLM instance from model identifier.

    Args:
        config: Codur configuration
        model: Model name to look up, or None to use default
        temperature: Optional temperature override
        json_mode: Whether to enable JSON mode

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If no default LLM profile is configured
    """
    matching_profile = None
    if model:
        for profile_name, profile in config.llm.profiles.items():
            if profile.model == model:
                matching_profile = profile_name
                break

    profile_to_use = matching_profile if matching_profile else config.llm.default_profile
    if not profile_to_use:
        raise ValueError("No default LLM profile configured.")

    return create_llm_profile(config, profile_to_use, temperature=temperature, json_mode=json_mode)
