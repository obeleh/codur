"""Tests for non-LLM tool routing."""

from langchain_core.messages import HumanMessage

from codur.graph.non_llm_tools import run_non_llm_tools


def _make_state(messages):
    """Create state with messages."""
    return {"messages": messages}


def test_non_llm_tools_skips_code_fix_with_file_hint():
    messages = [HumanMessage(content="fix the bug in @main.py. The total is wrong.")]
    state = _make_state(messages)
    result = run_non_llm_tools(messages, state)
    assert result is None


def test_non_llm_tools_allows_file_delete():
    messages = [HumanMessage(content="delete @old.py")]
    state = _make_state(messages)
    result = run_non_llm_tools(messages, state)
    assert result is not None
    assert result["next_action"] == "tool"
    assert result["tool_calls"][0]["tool"] == "delete_file"


def test_non_llm_tools_allows_read_file():
    messages = [HumanMessage(content="read README.md")]
    state = _make_state(messages)
    result = run_non_llm_tools(messages, state)
    assert result is not None
    tools = [call["tool"] for call in result["tool_calls"]]
    assert "read_file" in tools
