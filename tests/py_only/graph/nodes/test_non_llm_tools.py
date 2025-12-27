"""Tests for non-LLM tool routing."""

from langchain_core.messages import HumanMessage

from codur.graph.nodes.non_llm_tools import run_non_llm_tools


def test_non_llm_tools_skips_code_fix_with_file_hint():
    state = {}
    messages = [HumanMessage(content="fix the bug in @main.py. The total is wrong.")]
    result = run_non_llm_tools(messages, state)
    assert result is None


def test_non_llm_tools_allows_file_delete():
    state = {}
    messages = [HumanMessage(content="delete @old.py")]
    result = run_non_llm_tools(messages, state)
    assert result is not None
    assert result["next_action"] == "tool"
    assert result["tool_calls"][0]["tool"] == "delete_file"


def test_non_llm_tools_allows_read_file():
    state = {}
    messages = [HumanMessage(content="read README.md")]
    result = run_non_llm_tools(messages, state)
    assert result is not None
    tools = [call["tool"] for call in result["tool_calls"]]
    assert "read_file" in tools
