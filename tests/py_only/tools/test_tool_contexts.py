"""Tests for tool context annotations and injection."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from codur.config import CodurConfig, LLMSettings
from codur.graph.state import AgentStateData
from codur.graph.tool_executor import _build_tool_map
from codur.tools.agents import agent_call
from codur.tools.code_modification import replace_file_content
from codur.tools.filesystem import list_files, read_file, write_file
from codur.tools.tool_annotations import ToolContext, ToolGuard, get_tool_contexts, get_tool_guards


def test_tool_context_metadata():
    assert ToolContext.FILESYSTEM in get_tool_contexts(read_file)
    assert ToolContext.SEARCH in get_tool_contexts(list_files)
    assert ToolContext.CONFIG in get_tool_contexts(agent_call)
    assert ToolGuard.TEST_OVERWRITE in get_tool_guards(write_file)
    assert ToolGuard.TEST_OVERWRITE in get_tool_guards(replace_file_content)


def test_tool_executor_injects_contexts(tmp_path: Path):
    sample_path = tmp_path / "sample.txt"
    sample_path.write_text("hello", encoding="utf-8")

    config = CodurConfig(llm=LLMSettings(default_profile="test"))
    state = AgentStateData({"config": config, "messages": [HumanMessage(content="test")]})
    tool_map = _build_tool_map(tmp_path, False, state, None, config)

    output = tool_map["read_file"]({"path": "sample.txt"})
    assert output == "hello"

    files = tool_map["list_files"]({})
    assert "sample.txt" in files
