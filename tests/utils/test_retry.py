import pytest

from langchain_core.messages import AIMessage

from codur.config import CodurConfig, LLMSettings, RuntimeSettings
from codur.utils.retry import LLMRetryStrategy, RetryStrategy, retry_with_backoff


def test_retry_with_backoff_succeeds_after_failure() -> None:
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise Exception("connection error")
        return AIMessage(content="ok")

    strategy = RetryStrategy(max_attempts=3, initial_delay=0)
    result = retry_with_backoff(flaky, strategy)
    assert result.content == "ok"


def test_retry_with_backoff_stops_on_non_retryable() -> None:
    def failing():
        raise Exception("bad request")

    strategy = RetryStrategy(max_attempts=3, initial_delay=0, retry_on=lambda exc: False)
    with pytest.raises(Exception, match="bad request"):
        retry_with_backoff(failing, strategy)


def test_llm_retry_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    config = CodurConfig(
        llm=LLMSettings(default_profile="primary"),
        runtime=RuntimeSettings(planner_fallback_profiles=["fallback"]),
    )

    class StubLLM:
        def __init__(self, error: Exception | None = None):
            self.error = error

        def invoke(self, prompt_messages):
            if self.error:
                raise self.error
            return AIMessage(content="ok")

    primary = StubLLM(error=Exception("connection error"))
    fallback = StubLLM()

    def fake_create_llm_profile(cfg, profile_name):
        assert profile_name == "fallback"
        return fallback

    monkeypatch.setattr("codur.utils.retry.create_llm_profile", fake_create_llm_profile)

    retry = LLMRetryStrategy(max_attempts=1, initial_delay=0)
    llm, response, profile_name = retry.invoke_with_fallbacks(config, primary, [])
    assert llm is fallback
    assert profile_name == "fallback"
    assert response.content == "ok"


def test_llm_retry_with_fallback_rethrows_non_connection() -> None:
    config = CodurConfig(
        llm=LLMSettings(default_profile="primary"),
        runtime=RuntimeSettings(planner_fallback_profiles=["fallback"]),
    )

    class StubLLM:
        def invoke(self, prompt_messages):
            raise Exception("bad request")

    retry = LLMRetryStrategy(max_attempts=1, initial_delay=0)
    with pytest.raises(Exception, match="bad request"):
        retry.invoke_with_fallbacks(config, StubLLM(), [])
