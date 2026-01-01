"""Shared helpers for creating and invoking LLMs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state_operations import is_verbose, get_messages
from codur.graph.tool_executor import execute_tool_calls, ToolExecutionResult
from codur.llm import create_llm, create_llm_profile
from codur.utils.llm_calls import invoke_llm
from codur.utils.message_pipeline import message_shortening_pipeline
from codur.utils.tool_response_handler import deserialize_tool_calls, extract_tool_calls_from_json_text

if TYPE_CHECKING:
    from codur.graph.state import AgentState

console = Console()


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


def is_retryable_error(error_message: str) -> bool:
    if "Tool call validation failed: tool call validation failed" in error_message:
        return True
    if "Parsing failed. The model generated output that could not be parsed" in error_message:
        return True
    if "Tool call validation failed: tool call validation failed" in error_message:
        return True
    return False


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
    tool_names = [schema["name"] for schema in tool_schemas]

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
        messages_for_llm = message_shortening_pipeline(get_messages(state) + new_messages, tool_names)

        try:
            response = invoke_llm(
                llm,
                messages_for_llm,
                invoked_by=invoked_by,
                state=state,
                config=config,
            )
        except Exception as e:
            if is_retryable_error(str(e)):
                console.log("[yellow]Retrying LLM invocation due to retryable error...[/yellow]")
                response = invoke_llm(
                    llm,
                    messages_for_llm,
                    invoked_by=invoked_by,
                    state=state,
                    config=config,
                )
            else:
                raise

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
        messages_for_llm = message_shortening_pipeline(get_messages(state) + new_messages, tool_names)

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

    execution_result = execute_tool_calls(tool_calls, state, config, augment=False, summary_mode="brief")
    new_messages.extend(execution_result.messages)

    if tool_calls:
        _checkup_consecutive_toolcalls(new_messages, state, tool_calls)

    return new_messages, execution_result


def _checkup_consecutive_toolcalls(
    new_messages: list[BaseMessage],
    state: AgentState | None,
    tool_calls: list[dict]
) -> None:
    # When the same tool was called in the previous iteration, prompt for next steps to avoid loops
    current_tool_names = {tc.get("tool") for tc in tool_calls}
    previous_messages = get_messages(state)
    # Find the most recent tool messages from the previous iteration
    previous_tool_names = set()
    for msg in reversed(previous_messages):
        if isinstance(msg, ToolMessage):
            previous_tool_names.add(msg.tool_name)
        elif previous_tool_names:
            # Stop when we hit a non-tool message after finding tool messages
            break

    repeated_tools = current_tool_names & previous_tool_names
    if repeated_tools:
        tools_str = ", ".join(repeated_tools)
        msg = f"You called the same tool(s) again: {tools_str}. Please proceed with the next step or explain what you're trying to achieve. Used the clarify tool to express your intent."
        if is_verbose(state):
            console.log(f"[yellow]Detected repeated tool calls: {tools_str}[/yellow]")
        # last_msg = new_messages.pop()
        new_messages.append(HumanMessage(content=msg))
        # new_messages.append(last_msg)


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
