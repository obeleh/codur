"""Shared helpers for graph nodes."""

from typing import Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from codur.config import CodurConfig


def _resolve_agent_reference(raw_agent: str) -> str:
    """Extract the actual agent name from agent reference format (e.g., 'agent:ollama' -> 'ollama')."""
    if raw_agent.startswith("agent:"):
        return raw_agent.split(":", 1)[1]
    return raw_agent


def _resolve_agent_profile(config: CodurConfig, agent_name: str) -> tuple[str, Optional[dict]]:
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


def _normalize_messages(messages: Any) -> list[BaseMessage]:
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
