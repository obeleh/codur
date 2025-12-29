"""Shared helpers for creating and invoking LLMs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state_operations import is_verbose, get_messages
from codur.graph.tool_executor import execute_tool_calls, ToolExecutionResult
from codur.llm import create_llm, create_llm_profile
from codur.utils.llm_calls import invoke_llm
from codur.utils.tool_response_handler import deserialize_tool_calls, extract_tool_calls_from_json_text

if TYPE_CHECKING:
    from codur.graph.state import AgentState

console = Console()


class ShortenableSystemMessage(SystemMessage):
    """System message with a shorter summary for LLM calls."""

    short_content: str | None = None
    long_form_visible_for_agent_name: str | None = None


def message_shortening_pipeline(messages: list[BaseMessage], only_for_agent: str | None = None) -> list[BaseMessage]:
    """Return messages optimized for LLM calls, shortening selected system messages.

    Rules:
    - If only_for_agent is specified and matches message's agent, shorten it
    - If this is not the last ShortenableSystemMessage, shorten it
    - Otherwise, keep the full content (especially important for the last system message)
    """
    shortened: list[BaseMessage] = []
    last_shortenable_index = None
    for idx, message in enumerate(messages):
        if isinstance(message, ShortenableSystemMessage):
            last_shortenable_index = idx

    for idx, message in enumerate(messages):
        if isinstance(message, ShortenableSystemMessage):
            # Rule 1: If only_for_agent matches, shorten it
            if only_for_agent and message.long_form_visible_for_agent_name == only_for_agent:
                short_msg = SystemMessage(content=message.short_content)
                shortened.append(short_msg)
                continue
            # Rule 2: If this is not the last shortenable message and we have short content, shorten it
            elif message.short_content and idx != last_shortenable_index:
                short_msg = SystemMessage(content=message.short_content)
                shortened.append(short_msg)
                continue

        # Otherwise keep the message as-is
        shortened.append(message)
    return shortened


def _profile_for_model(config: CodurConfig, model: str | None) -> str | None:
    if not model:
        return None
    for profile_name, profile in config.llm.profiles.items():
        if profile.model == model:
            return profile_name
    return None


def _create_llm(
    config: CodurConfig,
    profile_name: str | None,
    json_mode: bool,
    temperature: float | None,
) -> BaseChatModel:
    if profile_name:
        return create_llm_profile(
            config,
            profile_name,
            json_mode=json_mode,
            temperature=temperature,
        )
    if json_mode:
        if not config.llm.default_profile:
            raise ValueError("No default LLM profile configured (llm.default_profile)")
        return create_llm_profile(
            config,
            config.llm.default_profile,
            json_mode=json_mode,
            temperature=temperature,
        )
    return create_llm(config, temperature=temperature)


def _is_json_validation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "json_validate_failed" in message
        or "failed to validate json" in message
        or ("json" in message and "400" in message)
    )


def create_and_invoke(
    config: CodurConfig,
    messages: list[BaseMessage],
    profile_name: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    invoked_by: str = "unknown",
    state: "AgentState | None" = None,
) -> BaseMessage:
    """Create an LLM and invoke it, returning the response message."""
    llm = _create_llm(
        config,
        profile_name=profile_name,
        json_mode=json_mode,
        temperature=temperature,
    )
    return invoke_llm(
        llm,
        messages,
        invoked_by=invoked_by,
        state=state,
        config=config,
    )


def create_and_invoke_with_tool_support(
    config: CodurConfig,
    new_messages: list[BaseMessage],
    tool_schemas: list[dict],
    profile_name: str | None = None,
    temperature: float | None = None,
    invoked_by: str = "unknown",
    state: "AgentState | None" = None,
) -> tuple[list[BaseMessage], ToolExecutionResult]:
    """
    Invoke LLM with tools if supported, else JSON fallback.

    Automatically detects provider capability:
    - If provider supports native tools: bind tools and invoke
    - If provider doesn't support native tools: use json_mode with prompt injection

    Args:
        config: Codur configuration
        new_messages: Conversation messages
        tool_schemas: Tool schemas to bind
        profile_name: Profile name from config
        temperature: Override temperature
        invoked_by: Caller identifier for logging
        state: Agent state
    """
    from codur.providers.base import ProviderRegistry
    from codur.llm import create_llm_with_tools

    # Resolve profile
    resolved_profile = profile_name or config.llm.default_profile
    if not resolved_profile:
        raise ValueError("No profile specified and no default profile configured")

    profile = config.llm.profiles.get(resolved_profile)
    if not profile:
        raise ValueError(f"Profile '{resolved_profile}' not found")

    provider_name = profile.provider
    provider_class = ProviderRegistry.get(provider_name)

    verbose = is_verbose(state)
    agent_name = invoked_by.split(".")[0] if "." in invoked_by else invoked_by

    # Check if provider supports native tool calling
    if provider_class.supports_native_tools():
        if verbose:
            console.log("[bold cyan]LLM with native tools...[/bold cyan]")

        # Native path: bind tools and invoke
        llm = create_llm_with_tools(
            config,
            resolved_profile,
            tool_schemas,
            temperature=temperature,
        )
        messages_for_llm = message_shortening_pipeline(get_messages(state) + new_messages, only_for_agent=agent_name)
        response = invoke_llm(
            llm,
            messages_for_llm,
            invoked_by=invoked_by,
            state=state,
            config=config,
        )
        response_dict = deserialize_tool_calls(response)
        tool_calls = response_dict["tool_calls"]
        new_messages.append(AIMessage(content=json.dumps(response_dict)))
    else:
        if verbose:
            console.log("[bold cyan]LLM with json fallback tools...[/bold cyan]")

        # JSON fallback path: inject tools in prompt, use json_mode
        # Build tool descriptions for prompt
        tool_descriptions = _build_tool_descriptions_for_prompt(tool_schemas)
        system_message_content = (
            f"Available tools:\n{tool_descriptions}\n\n"
            "Return JSON with: {\"thought\": \"...\", \"tool_calls\": [{\"tool\": \"name\", \"args\": {...}}]}"
        )

        # Prepend system message
        new_messages = [SystemMessage(content=system_message_content)] + list(new_messages)
        messages_for_llm = message_shortening_pipeline(get_messages(state) + new_messages, only_for_agent=agent_name)

        # Create LLM with json_mode
        llm = _create_llm(
            config,
            resolved_profile,
            json_mode=True,
            temperature=temperature,
        )
        response = invoke_llm(
            llm,
            messages_for_llm,
            invoked_by=f"{invoked_by}.json_fallback",
            state=state,
            config=config,
        )
        tool_calls = extract_tool_calls_from_json_text(response)
        new_messages.append(AIMessage(content=response.content))

    # TODO: filter the available tools for execute_tool_calls?
    execution_result = execute_tool_calls(tool_calls, state, config, augment=False, summary_mode="brief")
    new_messages.extend(execution_result.messages)
    return new_messages, execution_result

def _build_tool_descriptions_for_prompt(tool_schemas: list[dict]) -> str:
    """Build tool descriptions for prompt injection (JSON fallback)."""
    lines = []
    for schema in tool_schemas[:20]:  # Limit to prevent prompt bloat
        name = schema["name"]
        desc = schema.get("description", "")
        params = schema.get("parameters", {}).get("properties", {})
        args_str = ", ".join([f'"{k}": "{v.get("type", "string")}"' for k, v in params.items()])
        lines.append(f'{{"tool": "{name}", "args": {{{args_str}}}}}  # {desc}')
    return "\n".join(lines)
