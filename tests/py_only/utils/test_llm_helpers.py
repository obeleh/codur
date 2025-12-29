"""Tests for LLM helper utilities."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from codur.utils.llm_helpers import ShortenableSystemMessage, message_shortening_pipeline


def test_message_shortening_pipeline_keeps_latest_system_message():
    messages = [
        ShortenableSystemMessage(content="full-1", short_content="short-1"),
        HumanMessage(content="first task"),
        ShortenableSystemMessage(content="full-2", short_content="short-2"),
        HumanMessage(content="second task"),
    ]

    shortened = message_shortening_pipeline(messages)

    assert isinstance(shortened[0], SystemMessage)
    assert shortened[0].content == "short-1"
    assert isinstance(shortened[2], ShortenableSystemMessage)
    assert shortened[2].content == "full-2"


def test_message_shortening_pipeline_does_not_shorten_single_system_message():
    messages = [
        ShortenableSystemMessage(content="full-only", short_content="short-only"),
        HumanMessage(content="task"),
    ]

    shortened = message_shortening_pipeline(messages)

    assert isinstance(shortened[0], ShortenableSystemMessage)
    assert shortened[0].content == "full-only"
