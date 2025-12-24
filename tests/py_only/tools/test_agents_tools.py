"""Tests for agent tools."""

import pytest

from codur.tools.agents import agent_call, retry_in_agent


def test_agent_call_requires_agent_and_challenge():
    with pytest.raises(ValueError, match="agent_call requires an 'agent'"):
        agent_call("", "task")
    with pytest.raises(ValueError, match="agent_call requires a 'challenge'"):
        agent_call("codex", "")


def test_agent_call_requires_config():
    with pytest.raises(ValueError, match="Config not available"):
        agent_call("codex", "do the thing")


def test_retry_in_agent_requires_agent_and_task():
    with pytest.raises(ValueError, match="retry_in_agent requires an 'agent'"):
        retry_in_agent("", "task")
    with pytest.raises(ValueError, match="retry_in_agent requires a 'task'"):
        retry_in_agent("codex", "")


def test_retry_in_agent_requires_config():
    with pytest.raises(ValueError, match="Config not available"):
        retry_in_agent("codex", "task")
