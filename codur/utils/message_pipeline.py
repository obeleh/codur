"""Message pipeline utilities for context management."""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from codur.graph.state_operations import parse_tool_message
from codur.tools.registry import get_tools_with_side_effects


def _format_tool_call(msg: ToolMessage, max_args_len: int = 200) -> str:
    """Format a tool call with truncated args."""
    assert isinstance(msg, ToolMessage)
    parsed = parse_tool_message(msg)
    if parsed:
        args_str = json.dumps(parsed.args)
        if len(args_str) > max_args_len:
            args_str = args_str[:max_args_len] + "..."
        return f"- {parsed.tool} with args {args_str}"
    return f"- {msg.tool_name or 'unknown'}"


def message_shortening_pipeline(messages: list[BaseMessage], known_tool_names: list[str] | None=None) -> list[BaseMessage]:
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
            # filter out tools that would be unknown to the LLM
            if known_tool_names and message.tool_name not in known_tool_names:
                continue
            tool_calls_to_inject.append(message)

    system_message = messages[last_system_idx]
    messages_after_system = tool_calls_to_inject + messages[last_system_idx + 1:]

    formatted_tool_calls = []
    for msg in messages_after_system:
        if isinstance(msg, ToolMessage):
            formatted_tool_calls.append(_format_tool_call(msg))

    do_not_recall_tool_msg = (
        f"Called tools in context:\n{chr(10).join(formatted_tool_calls)}\n\n"
        "GOOD: Use the tool results already in context to inform your response.\n"
        "BAD: Calling the same tool again with identical arguments.\n\n"
        "Only re-call a tool if the arguments have changed. "
        "If you need to re-call with the same arguments, use the \"clarify\" tool first to explain why."
    )

    system_message = SystemMessage(
        content=system_message.content + "\n\n" + do_not_recall_tool_msg
    )

    # Output: system message + preserved tool calls + messages after system message
    messages_out = [system_message] + messages_after_system
    return messages_out