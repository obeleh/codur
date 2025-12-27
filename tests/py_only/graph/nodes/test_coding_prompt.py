"""Tests for the codur-coding system prompt."""

from codur.graph.nodes.coding import CODING_AGENT_SYSTEM_PROMPT


def test_coding_prompt_includes_rope_tools():
    assert "rope_find_usages" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_find_definition" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_rename_symbol" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_move_module" in CODING_AGENT_SYSTEM_PROMPT
    assert "rope_extract_method" in CODING_AGENT_SYSTEM_PROMPT
