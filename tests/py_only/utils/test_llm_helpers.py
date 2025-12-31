"""Tests for LLM helper utilities."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from codur.utils.message_pipeline import message_shortening_pipeline


def test_message_shortening_pipeline_keeps_only_last_system_message():
    """Drops everything before the last SystemMessage."""
    messages = [
        SystemMessage(content="full-1"),
        HumanMessage(content="first task"),
        SystemMessage(content="full-2"),
        HumanMessage(content="second task"),
    ]

    shortened = message_shortening_pipeline(messages)

    # Only last SystemMessage + messages after it are kept
    assert len(shortened) == 2
    assert isinstance(shortened[0], SystemMessage)
    assert shortened[0].content == "full-2"
    assert isinstance(shortened[1], HumanMessage)
    assert shortened[1].content == "second task"


def test_message_shortening_pipeline_preserves_single_system_message():
    messages = [
        SystemMessage(content="only-system"),
        HumanMessage(content="task"),
    ]

    shortened = message_shortening_pipeline(messages)

    assert isinstance(shortened[0], SystemMessage)
    assert shortened[0].content == "only-system"
