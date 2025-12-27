"""Planning decision validators."""

from __future__ import annotations


def looks_like_change_request(text: str) -> bool:
    lowered = text.lower()
    triggers = ("fix", "edit", "update", "change", "modify", "refactor", "bug", "issue")
    return any(trigger in lowered for trigger in triggers)


def mentions_file_path(text: str) -> bool:
    return "@" in text or ".py" in text or "/" in text or "\\" in text


def has_mutation_tool(tool_calls: list[dict]) -> bool:
    mutating = {
        "write_file",
        "append_file",
        "replace_in_file",
        "delete_file",
        "copy_file",
        "move_file",
        "copy_file_to_dir",
        "move_file_to_dir",
        "write_json",
        "set_json_value",
        "write_yaml",
        "set_yaml_value",
        "write_ini",
        "set_ini_value",
        "inject_function",
        "replace_function",
        "replace_class",
        "replace_method",
        "replace_file_content",
        "rope_rename_symbol",
        "rope_move_module",
        "rope_extract_method",
    }
    for call in tool_calls or []:
        tool_name = call.get("tool")
        if tool_name in mutating:
            return True
    return False
