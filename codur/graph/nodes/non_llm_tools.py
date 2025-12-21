"""Non-LLM planning helpers that can directly produce actions."""

from __future__ import annotations

from typing import Optional
import os
import re
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from codur.graph.state import AgentState
from codur.graph.nodes.types import PlanNodeResult
from codur.tools.filesystem import EXCLUDE_DIRS

GREETING_MAX_WORDS = 3


def _trivial_response(text: str) -> Optional[str]:
    stripped = text.strip().lower()
    if not stripped:
        return "Hi! How can I help you with your coding tasks today?"
    if "thank" in stripped or "thanks" in stripped:
        return "You're welcome! Anything else you want to tackle?"
    greetings = {
        "hi", "hello", "hey", "yo", "sup",
        "good morning", "good afternoon", "good evening",
    }
    if stripped in greetings:
        return "Hi! How can I help you with your coding tasks today?"
    if len(stripped.split()) <= GREETING_MAX_WORDS and any(word in greetings for word in stripped.split()):
        return "Hi! How can I help you with your coding tasks today?"
    return None


def _looks_like_explain_request(text: str) -> bool:
    lowered = text.lower()
    triggers = ("what does", "explain", "describe", "summarize", "summary of")
    return any(trigger in lowered for trigger in triggers)


def _find_workspace_match(raw_text: str) -> Optional[str]:
    text = raw_text.strip()
    if not text:
        return None
    candidates = []
    for token in text.replace('"', " ").replace("'", " ").split():
        if token.startswith("@") or ".py" in token or "/" in token or "\\" in token:
            cleaned = token.strip(".,:;()[]{}")
            if cleaned.startswith("@"):
                cleaned = cleaned[1:]
            if cleaned:
                candidates.append(cleaned)
    cwd = Path.cwd()
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = cwd / path

        try:
            if path.is_absolute() and not path.resolve().is_relative_to(cwd.resolve()):
                continue
        except (ValueError, OSError):
            continue

        if path.exists():
            try:
                return str(path.relative_to(cwd) if path.is_relative_to(cwd) else path)
            except ValueError:
                continue

        if path.name == candidate and not ("/" in candidate or "\\" in candidate):
            matches = []
            for root, dirnames, filenames in os.walk(cwd):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
                if candidate in filenames:
                    matches.append(Path(root) / candidate)
                if len(matches) > 1:
                    break
            if len(matches) == 1:
                match = matches[0]
                try:
                    return str(match.relative_to(cwd))
                except ValueError:
                    continue
    return None


def _detect_tool_operations(message: str) -> Optional[list[dict]]:
    msg = message.strip()
    msg_lower = msg.lower()

    def _extract_quoted(text: str) -> Optional[str]:
        match = re.search(r"\"([^\"]+)\"|'([^']+)'", text)
        if not match:
            return None
        return match.group(1) or match.group(2)

    # File operations: move/copy/delete/read/write/append/line_count
    move_match = re.search(r"move\s+([^\s]+)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
    if move_match:
        source = move_match.group(1).strip()
        dest = move_match.group(2).strip()
        if dest.endswith("/") or (Path(dest).exists() and Path(dest).is_dir()):
            return [{"tool": "move_file_to_dir", "args": {"source": source, "destination_dir": dest.rstrip("/")}}]
        return [{"tool": "move_file", "args": {"source": source, "destination": dest}}]

    copy_match = re.search(r"copy\s+([^\s]+)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
    if copy_match:
        source = copy_match.group(1).strip()
        dest = copy_match.group(2).strip()
        if dest.endswith("/") or (Path(dest).exists() and Path(dest).is_dir()):
            return [{"tool": "copy_file_to_dir", "args": {"source": source, "destination_dir": dest.rstrip("/")}}]
        return [{"tool": "copy_file", "args": {"source": source, "destination": dest}}]

    delete_match = re.search(r"delete\s+([^\s]+)", msg, re.IGNORECASE)
    if delete_match:
        path = delete_match.group(1).strip()
        return [{"tool": "delete_file", "args": {"path": path}}]

    read_match = re.search(r"(?:read|show|open)\s+([^\s]+)", msg, re.IGNORECASE)
    if read_match:
        path = read_match.group(1).strip()
        return [{"tool": "read_file", "args": {"path": path}}]

    write_match = re.search(r"write\s+(.+?)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
    if write_match:
        content = _extract_quoted(write_match.group(1)) or write_match.group(1).strip()
        path = write_match.group(2).strip()
        return [{"tool": "write_file", "args": {"path": path, "content": content}}]

    append_match = re.search(r"append\s+(.+?)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
    if append_match:
        content = _extract_quoted(append_match.group(1)) or append_match.group(1).strip()
        path = append_match.group(2).strip()
        return [{"tool": "append_file", "args": {"path": path, "content": content}}]

    line_match = re.search(r"(?:line\s+count|lines)\s+(?:of|in)\s+([^\s]+)", msg, re.IGNORECASE)
    if line_match:
        path = line_match.group(1).strip()
        return [{"tool": "line_count", "args": {"path": path}}]

    # File discovery / search
    if re.search(r"list\s+files", msg_lower):
        dir_match = re.search(r"list\s+files\s+in\s+([^\s]+)", msg, re.IGNORECASE)
        args = {"root": dir_match.group(1).strip()} if dir_match else {}
        return [{"tool": "list_files", "args": args}]

    if re.search(r"list\s+dirs|list\s+directories", msg_lower):
        dir_match = re.search(r"list\s+(?:dirs|directories)\s+in\s+([^\s]+)", msg, re.IGNORECASE)
        args = {"root": dir_match.group(1).strip()} if dir_match else {}
        return [{"tool": "list_dirs", "args": args}]

    search_match = re.search(r"(?:find|search)\s+files?\s+(?:named|for)\s+(.+)", msg, re.IGNORECASE)
    if search_match:
        query = _extract_quoted(search_match.group(1)) or search_match.group(1).strip()
        return [{"tool": "search_files", "args": {"query": query}}]

    grep_match = re.search(r"(?:grep|search)\s+for\s+(.+?)\s+in\s+([^\s]+)", msg, re.IGNORECASE)
    if grep_match:
        pattern = _extract_quoted(grep_match.group(1)) or grep_match.group(1).strip()
        root = grep_match.group(2).strip()
        return [{"tool": "grep_files", "args": {"pattern": pattern, "root": root}}]

    replace_match = re.search(r"replace\s+(.+?)\s+with\s+(.+?)\s+in\s+([^\s]+)", msg, re.IGNORECASE)
    if replace_match:
        pattern = _extract_quoted(replace_match.group(1)) or replace_match.group(1).strip()
        replacement = _extract_quoted(replace_match.group(2)) or replace_match.group(2).strip()
        path = replace_match.group(3).strip()
        return [{"tool": "replace_in_file", "args": {"path": path, "pattern": pattern, "replacement": replacement}}]

    # Structured data (json/yaml/ini)
    read_struct_match = re.search(r"read\s+(json|yaml|yml|ini)\s+([^\s]+)", msg, re.IGNORECASE)
    if read_struct_match:
        fmt = read_struct_match.group(1).lower()
        path = read_struct_match.group(2).strip()
        tool = {"json": "read_json", "yaml": "read_yaml", "yml": "read_yaml", "ini": "read_ini"}[fmt]
        return [{"tool": tool, "args": {"path": path}}]

    write_struct_match = re.search(r"write\s+(json|yaml|yml|ini)\s+(.+?)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
    if write_struct_match:
        fmt = write_struct_match.group(1).lower()
        content = _extract_quoted(write_struct_match.group(2)) or write_struct_match.group(2).strip()
        path = write_struct_match.group(3).strip()
        tool = {"json": "write_json", "yaml": "write_yaml", "yml": "write_yaml", "ini": "write_ini"}[fmt]
        return [{"tool": tool, "args": {"path": path, "data": content}}]

    set_struct_match = re.search(
        r"set\s+(json|yaml|yml|ini)\s+(.+?)\s+in\s+([^\s]+)\s+to\s+(.+)",
        msg,
        re.IGNORECASE,
    )
    if set_struct_match:
        fmt = set_struct_match.group(1).lower()
        key_path = _extract_quoted(set_struct_match.group(2)) or set_struct_match.group(2).strip()
        path = set_struct_match.group(3).strip()
        value = _extract_quoted(set_struct_match.group(4)) or set_struct_match.group(4).strip()
        tool = {
            "json": "set_json_value",
            "yaml": "set_yaml_value",
            "yml": "set_yaml_value",
            "ini": "set_ini_value",
        }[fmt]
        if tool == "set_ini_value":
            section_option = key_path.split(".", 1)
            if len(section_option) != 2:
                return None
            args = {"path": path, "section": section_option[0], "option": section_option[1], "value": value}
        else:
            args = {"path": path, "key_path": key_path, "value": value}
        return [{"tool": tool, "args": args}]

    # Linting
    lint_tree_match = re.search(r"lint\s+(?:python\s+)?tree\s+([^\s]+)?", msg, re.IGNORECASE)
    if lint_tree_match:
        root = lint_tree_match.group(1).strip() if lint_tree_match.group(1) else None
        args = {"root": root} if root else {}
        return [{"tool": "lint_python_tree", "args": args}]

    lint_files_match = re.search(r"lint\s+(.+)", msg, re.IGNORECASE)
    if lint_files_match:
        raw = lint_files_match.group(1)
        paths = re.findall(r"[^\s]+\.py", raw)
        if paths:
            return [{"tool": "lint_python_files", "args": {"paths": paths}}]

    return None


def run_non_llm_tools(messages: list[BaseMessage], state: AgentState) -> Optional[PlanNodeResult]:
    tool_results_present = any(
        isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
        for msg in messages
    )

    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if last_human_msg:
        trivial_response = _trivial_response(last_human_msg)
        if trivial_response:
            return {
                "next_action": "end",
                "final_response": trivial_response,
                "iterations": state.get("iterations", 0) + 1,
            }

    if last_human_msg and not tool_results_present:
        if _looks_like_explain_request(last_human_msg):
            matched_path = _find_workspace_match(last_human_msg)
            if matched_path:
                return {
                    "next_action": "tool",
                    "tool_calls": [{"tool": "read_file", "args": {"path": matched_path}}],
                    "iterations": state.get("iterations", 0) + 1,
                }

        file_op_result = _detect_tool_operations(last_human_msg)
        if file_op_result:
            return {
                "next_action": "tool",
                "tool_calls": file_op_result,
                "iterations": state.get("iterations", 0) + 1,
            }

    return None
