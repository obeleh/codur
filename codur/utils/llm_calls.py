"""
LLM invocation accounting and limits.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from codur.config import CodurConfig
from codur.graph.state import AgentState
from rich.console import Console

console = Console()

class LLMCallLimitExceeded(RuntimeError):
    """Raised when the LLM call limit is exceeded."""


def _sanitize_tool_messages(llm: BaseChatModel, messages: list[BaseMessage]) -> list[BaseMessage]:
    """Convert ToolMessages to SystemMessages if LLM doesn't have tools bound.

    When an LLM is called without bound tools (e.g., planning with json_mode),
    any ToolMessages from previous calls will cause errors with providers like Groq.
    This function converts those ToolMessages to SystemMessages to preserve information
    while avoiding API errors.

    Args:
        llm: The LLM instance to check for bound tools
        messages: List of messages to sanitize

    Returns:
        Sanitized list of messages
    """
    # Check if LLM has tools bound by looking for tool schemas
    # When tools are bound via bind_tools(), the LLM is wrapped in a RunnableBinding
    # with kwargs containing 'tools'
    has_bound_tools = False
    if hasattr(llm, 'kwargs') and 'tools' in llm.kwargs:
        bound_tools = llm.kwargs.get('tools', [])
        has_bound_tools = bool(bound_tools)

    if has_bound_tools:
        # If tools are bound, filter ToolMessages to only include bound tools
        bound_tool_names = set()
        for tool in llm.kwargs.get('tools', []):
            if isinstance(tool, dict):
                bound_tool_names.add(tool.get('name'))

        sanitized = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                if msg.name and msg.name in bound_tool_names:
                    # Tool is bound, keep as ToolMessage
                    sanitized.append(msg)
                else:
                    # Tool not bound, convert to SystemMessage
                    sanitized.append(SystemMessage(content=f"Tool results from {msg.name}:\n{msg.content}"))
            else:
                sanitized.append(msg)
        return sanitized
    else:
        # No tools bound - convert all ToolMessages to SystemMessages
        sanitized = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                sanitized.append(SystemMessage(content=f"Tool results from {msg.name}:\n{msg.content}"))
            else:
                sanitized.append(msg)
        return sanitized


def _get_matching_instructions(config: CodurConfig | None, invoked_by: str) -> list[str]:
    """Get all matching model agent instructions for the given invoked_by value.

    Args:
        config: Codur configuration (may be None)
        invoked_by: The invoked_by string (e.g., "coding.primary", "planning.llm_plan")

    Returns:
        List of instruction strings that match the invoked_by value
    """
    if not config or not config.model_agent_instructions:
        return []

    instructions = []
    for instruction_config in config.model_agent_instructions:
        agent_pattern = instruction_config.agent.lower()
        invoked_by_lower = invoked_by.lower()

        # Match "all" or partial match in invoked_by string
        if agent_pattern == "all" or agent_pattern in invoked_by_lower:
            instructions.append(instruction_config.instruction)

    return instructions


def invoke_llm(
    llm: BaseChatModel,
    prompt_messages: list[BaseMessage],
    invoked_by: str,
    state: AgentState | None = None,
    config: CodurConfig | None = None,
) -> BaseMessage:
    _increment_llm_calls(state, config, invoked_by)

    # Get matching instructions and inject as system messages
    instructions = _get_matching_instructions(config, invoked_by)
    if instructions:
        # Prepend instructions as system messages
        instruction_messages = [
            SystemMessage(content=instruction)
            for instruction in instructions
        ]
        prompt_messages = instruction_messages + list(prompt_messages)

    # Convert ToolMessages to SystemMessages if LLM doesn't have tools bound
    # This prevents errors with providers like Groq when ToolMessages reference
    # tools that aren't in the currently bound tool set
    prompt_messages = _sanitize_tool_messages(llm, prompt_messages)

    return llm.invoke(prompt_messages)


def _increment_llm_calls(
    state: AgentState | None,
    config: CodurConfig | None,
    invoked_by: str,
) -> None:
    if state is None:
        return
    limit = _resolve_limit(state, config)
    count = int(state.get("llm_calls", 0)) + 1
    console.log(f"LLM call count: {count} (invoked_by={invoked_by})")
    state["llm_calls"] = count
    if limit is not None and count > limit:
        raise LLMCallLimitExceeded(
            f"LLM call limit exceeded ({count}/{limit}) at {invoked_by}"
        )


def _resolve_limit(state: AgentState, config: CodurConfig | None) -> int | None:
    if "max_llm_calls" in state and state["max_llm_calls"] is not None:
        return int(state["max_llm_calls"])
    if config and config.runtime.max_llm_calls is not None:
        return int(config.runtime.max_llm_calls)
    return None
