"""
Retry helpers for network- and LLM-bound operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from codur.config import CodurConfig
from codur.llm import create_llm_profile
from codur.utils.llm_calls import invoke_llm


def _is_connection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "connection error" in message
        or "failed to establish a new connection" in message
        or "max retries exceeded" in message
    )


@dataclass
class RetryStrategy:
    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0
    max_delay: Optional[float] = None
    retry_on: Callable[[Exception], bool] = _is_connection_error


def retry_with_backoff(
    func: Callable[[], BaseMessage],
    strategy: RetryStrategy,
) -> BaseMessage:
    last_error: Exception | None = None
    delay = strategy.initial_delay
    for attempt in range(1, strategy.max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= strategy.max_attempts or not strategy.retry_on(exc):
                raise
            time.sleep(delay)
            delay = delay * strategy.backoff_factor
            if strategy.max_delay is not None:
                delay = min(delay, strategy.max_delay)
    if last_error:
        raise last_error
    raise RuntimeError("Retry failed without exception")


class LLMRetryStrategy:
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_delay: Optional[float] = None,
    ) -> None:
        self.strategy = RetryStrategy(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
        )

    def invoke_with_retries(
        self,
        llm: BaseChatModel,
        prompt_messages: list[BaseMessage],
        state: "AgentState | None" = None,
        config: CodurConfig | None = None,
        invoked_by: str = "retry.invoke_with_retries",
    ) -> BaseMessage:
        return retry_with_backoff(
            lambda: invoke_llm(
                llm,
                prompt_messages,
                invoked_by=invoked_by,
                state=state,
                config=config,
            ),
            self.strategy,
        )

    def invoke_with_fallbacks(
        self,
        config: CodurConfig,
        llm: BaseChatModel,
        prompt_messages: list[BaseMessage],
        state: "AgentState | None" = None,
        invoked_by: str = "retry.invoke_with_fallbacks",
    ) -> tuple[BaseChatModel, BaseMessage, str]:
        profile_names = [config.llm.default_profile]
        fallback_profiles = config.runtime.planner_fallback_profiles
        if fallback_profiles:
            profile_names += [name for name in fallback_profiles if name != config.llm.default_profile]

        last_error: Exception | None = None
        for idx, profile_name in enumerate(profile_names):
            try_llm = llm if idx == 0 else create_llm_profile(config, profile_name)
            try:
                response = self.invoke_with_retries(
                    try_llm,
                    prompt_messages,
                    state=state,
                    config=config,
                    invoked_by=invoked_by,
                )
                return try_llm, response, profile_name
            except Exception as exc:
                last_error = exc
                if not self.strategy.retry_on(exc):
                    raise
        if last_error:
            raise last_error
        raise RuntimeError("Planner LLM invocation failed without exception")
