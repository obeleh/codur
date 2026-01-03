"""Message pipeline utilities for context management."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage


def message_shortening_pipeline(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Shorten message history by keeping only the last SystemMessage and subsequent messages."""

    # Find the last SystemMessage index
    last_system_idx = None
    for idx, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            last_system_idx = idx

    if last_system_idx is None:
        raise ValueError("No SystemMessage found in messages")

    return messages[last_system_idx:]