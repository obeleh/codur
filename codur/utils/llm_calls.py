"""
LLM invocation accounting and limits.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from codur.config import CodurConfig
from codur.graph.state import AgentState
from rich.console import Console

console = Console()

class LLMCallLimitExceeded(RuntimeError):
    """Raised when the LLM call limit is exceeded."""


def invoke_llm(
    llm: BaseChatModel,
    prompt_messages: list[BaseMessage],
    invoked_by: str,
    state: AgentState | None = None,
    config: CodurConfig | None = None,
) -> BaseMessage:
    _increment_llm_calls(state, config, invoked_by)
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
