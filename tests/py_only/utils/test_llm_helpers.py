"""Tests for LLM helper utilities."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from codur.utils.llm_helpers import ShortenableSystemMessage, message_shortening_pipeline


def test_message_shortening_pipeline_keeps_only_last_shortenable():
    """New implementation drops everything before the last ShortenableSystemMessage."""
    messages = [
        ShortenableSystemMessage(content="full-1", short_content="short-1"),
        HumanMessage(content="first task"),
        ShortenableSystemMessage(content="full-2", short_content="short-2"),
        HumanMessage(content="second task"),
    ]

    shortened = message_shortening_pipeline(messages)

    # Only last shortenable + messages after it are kept
    assert len(shortened) == 2
    assert isinstance(shortened[0], ShortenableSystemMessage)
    assert shortened[0].content == "full-2"
    assert isinstance(shortened[1], HumanMessage)
    assert shortened[1].content == "second task"


def test_message_shortening_pipeline_does_not_shorten_single_system_message():
    messages = [
        ShortenableSystemMessage(content="full-only", short_content="short-only"),
        HumanMessage(content="task"),
    ]

    shortened = message_shortening_pipeline(messages)

    assert isinstance(shortened[0], ShortenableSystemMessage)
    assert shortened[0].content == "full-only"
