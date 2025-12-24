"""Tests for tool registry utilities."""

from codur.tools.registry import list_tool_directory, get_tool_help


def test_list_tool_directory_includes_core_tools():
    items = list_tool_directory()
    names = {item["name"] for item in items}
    assert "read_file" in names
    entry = next(item for item in items if item["name"] == "read_file")
    assert "root" not in entry["signature"]
    assert "state" not in entry["signature"]


def test_get_tool_help_unknown():
    result = get_tool_help("does_not_exist")
    assert "error" in result


def test_get_tool_help_signature_hides_internal_params():
    result = get_tool_help("read_file")
    assert "root" not in result["signature"]
    assert "allow_outside_root" not in result["signature"]
