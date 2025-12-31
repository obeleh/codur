"""Message pipeline utilities for context management."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from codur.tools.registry import get_tools_with_side_effects


def message_shortening_pipeline(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Returns all tool calls that were _after_ a mutation tool call.
    Mutation tool calls: ToolSideEffect.FILE_MUTATION and ToolSideEffect.STATE_CHANGE

    Only a single "SystemMessage" should be returned, everything before that (except the tool calls) is dropped

    The goal is to move the tool calls into the context, and drop everything before the last mutation,
    while keeping the last SystemMessage (which should contain important context).
    """
    # Get tool names with mutation side effects
    tools_with_side_effects = set(get_tools_with_side_effects())

    last_mutation_index = -1
    for idx, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            if message.name in tools_with_side_effects:
                last_mutation_index = idx

    # Find the last SystemMessage index
    last_system_idx = None
    for idx, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            last_system_idx = idx

    if last_system_idx is None:
        raise ValueError("No SystemMessage found in messages")

    # Determine which tool calls to preserve and inject after the system message
    # - If mutation before system message: collect from (mutation + 1) to system message
    # - If no mutation (or mutation after system message): collect all from 0 to system message
    if last_mutation_index != -1 and last_mutation_index < last_system_idx:
        start_idx = last_mutation_index + 1
    else:
        start_idx = 0

    tool_calls_to_inject = []
    for idx in range(start_idx, last_system_idx):
        message = messages[idx]
        if isinstance(message, ToolMessage):
            tool_calls_to_inject.append(message)

    system_message = messages[last_system_idx]

    # Output: system message + preserved tool calls + messages after system message
    messages_out = [system_message] + tool_calls_to_inject + messages[last_system_idx + 1:]
    return messages_out