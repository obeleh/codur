"""Tests for tool registry utilities."""

from codur.constants import TaskType
from codur.tools.registry import list_tool_directory, get_tool_help, list_tools_for_tasks


def test_list_tool_directory_includes_core_tools():
    items = list_tool_directory()
    names = {item["name"] for item in items}
    assert "read_file" in names
    entry = next(item for item in items if item["name"] == "read_file")
    assert "root" not in entry["signature"]
    assert "state" not in entry["signature"]


def test_list_tool_directory_hides_ripgrep_when_missing(monkeypatch):
    monkeypatch.setattr("codur.tools.registry._rg_available", lambda: False)
    items = list_tool_directory()
    names = {item["name"] for item in items}
    assert "ripgrep_search" not in names
    assert "grep_files" in names


def test_get_tool_help_unknown():
    result = get_tool_help("does_not_exist")
    assert "error" in result


def test_get_tool_help_signature_hides_internal_params():
    result = get_tool_help("read_file")
    assert "root" not in result["signature"]
    assert "allow_outside_root" not in result["signature"]


def test_list_tools_for_tasks_includes_replace_function_for_code_generation():
    items = list_tools_for_tasks([TaskType.CODE_GENERATION])
    names = {item["name"] for item in items}
    assert "replace_function" in names
