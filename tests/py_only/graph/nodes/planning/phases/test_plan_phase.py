"""Tests for full LLM planning phase."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from codur.graph.planning.phases.plan_phase import llm_plan


def _tool_msg(tool: str, output, args: dict | None = None) -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"tool": tool, "output": output, "args": args or {}}),
        tool_call_id="test",
    )


@pytest.fixture
def config():
    config = MagicMock()
    config.verbose = False
    return config


def test_llm_plan_lists_files_on_change_request(config):
    state = {
        "messages": [HumanMessage(content="Fix the bug")],
        "iterations": 0,
        "verbose": False,
        "config": config,
    }

    result = llm_plan(config, MagicMock(), MagicMock(), MagicMock(), state, MagicMock())

    assert result["next_action"] == "tool"
    assert result["tool_calls"] == [{"tool": "list_files", "args": {}}]


def test_llm_plan_selects_file_from_list_files(config):
    state = {
        "messages": [
            HumanMessage(content="Fix the bug"),
            _tool_msg("list_files", ["main.py", "expected.txt"]),
        ],
        "iterations": 1,
        "verbose": False,
        "config": config,
    }

    result = llm_plan(config, MagicMock(), MagicMock(), MagicMock(), state, MagicMock())

    assert result["next_action"] == "tool"
    assert result["tool_calls"] == [{"tool": "read_file", "args": {"path": "main.py"}}]
