"""Shared helpers for creating and invoking LLMs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.llm import create_llm, create_llm_profile
from codur.utils.llm_calls import invoke_llm

if TYPE_CHECKING:
    from codur.graph.state import AgentState

console = Console()


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


def invoke_llm_with_fallback(
    config: CodurConfig,
    messages: list[BaseMessage],
    profile_name: str | None = None,
    fallback_profile: str | None = None,
    fallback_model: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    invoked_by: str = "unknown",
    fallback_invoked_by: str | None = None,
    state: "AgentState | None" = None,
) -> tuple[BaseChatModel, BaseMessage]:
    """Create and invoke an LLM with fallback on JSON validation errors."""
    llm = _create_llm(
        config,
        profile_name=profile_name,
        json_mode=json_mode,
        temperature=temperature,
    )
    try:
        response = invoke_llm(
            llm,
            messages,
            invoked_by=invoked_by,
            state=state,
            config=config,
        )
        return llm, response
    except Exception as exc:
        if not _is_json_validation_error(exc):
            raise

    resolved_fallback_profile = fallback_profile
    if resolved_fallback_profile is None:
        resolved_fallback_profile = _profile_for_model(config, fallback_model)
    if resolved_fallback_profile is None:
        resolved_fallback_profile = config.llm.default_profile
    if not resolved_fallback_profile:
        raise ValueError("No fallback LLM profile configured (llm.default_profile)")

    fallback_label = fallback_profile or fallback_model or resolved_fallback_profile
    console.log(
        f"[yellow]JSON validation failed, trying fallback: {fallback_label}[/yellow]"
    )
    fallback_llm = _create_llm(
        config,
        profile_name=resolved_fallback_profile,
        json_mode=json_mode,
        temperature=temperature,
    )
    response = invoke_llm(
        fallback_llm,
        messages,
        invoked_by=fallback_invoked_by or f"{invoked_by}.fallback",
        state=state,
        config=config,
    )
    return fallback_llm, response
