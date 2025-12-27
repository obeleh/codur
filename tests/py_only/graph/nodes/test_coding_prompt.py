"""Tests for the codur-coding system prompt."""

from codur.graph.nodes.coding import CODING_AGENT_SYSTEM_PROMPT


def test_coding_system_prompt_includes_tools_section():
    """Test that the system prompt includes tools section."""
    assert "Available Tools" in CODING_AGENT_SYSTEM_PROMPT
    assert "File Operations" in CODING_AGENT_SYSTEM_PROMPT
    assert "Code Analysis & Modification" in CODING_AGENT_SYSTEM_PROMPT


def test_coding_prompt_includes_rope_tools():
    """Test that rope tools are included in the system prompt."""
    assert "rope_find_usages" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_find_definition" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_rename_symbol" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_move_module" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_extract_method" in CODING_AGENT_SYSTEM_PROMPT


def test_coding_prompt_includes_read_write_tools():
    """Test that file tools are included."""
    assert "read_file" in CODING_AGENT_SYSTEM_PROMPT
    assert "write_file" in CODING_AGENT_SYSTEM_PROMPT
    assert "replace_function" in CODING_AGENT_SYSTEM_PROMPT


def test_coding_prompt_warns_about_tool_invention():
    """Test that the prompt warns against inventing tools."""
    assert "Do NOT invent or create new tools" in CODING_AGENT_SYSTEM_PROMPT
    assert "Only use tools from this list" in CODING_AGENT_SYSTEM_PROMPT
