"""Registry-based tool detection for non-LLM planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Optional


ToolCallList = list[dict]


def _extract_quoted(text: str) -> Optional[str]:
    match = re.search(r"\"([^\"]+)\"|'([^']+)'", text)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _extract_path_from_message(text: str) -> Optional[str]:
    at_match = re.search(r"@([^\s,]+)", text)
    if at_match:
        return at_match.group(1)
    in_match = re.search(r"(?:in|inside)\s+([^\s,]+)", text, re.IGNORECASE)
    if in_match:
        return in_match.group(1)
    path_match = re.search(r"([^\s,]+\.py)", text)
    if path_match:
        return path_match.group(1)
    return None


@dataclass(frozen=True)
class ToolPattern:
    name: str
    detector: Callable[[str, str], Optional[ToolCallList]]
    priority: int = 0


class ToolDetector:
    def __init__(self) -> None:
        self._patterns: list[ToolPattern] = []

    def register(self, pattern: ToolPattern) -> None:
        self._patterns.append(pattern)
        self._patterns.sort(key=lambda item: item.priority, reverse=True)

    def detect(self, message: str) -> Optional[ToolCallList]:
        msg = message.strip()
        msg_lower = msg.lower()
        for pattern in self._patterns:
            result = pattern.detector(msg, msg_lower)
            if result:
                return result
        return None


def create_default_tool_detector() -> ToolDetector:
    detector = ToolDetector()

    def change_intent(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        change = re.search(r"\b(fix|edit|update|change|modify|refactor|bug|issue)\b", msg_lower)
        if not change:
            return None
        target = _extract_path_from_message(msg)
        if target:
            return [{"tool": "read_file", "args": {"path": target}}]
        return None

    def move_file(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"move\s+([^\s]+)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        source = match.group(1).strip()
        dest = match.group(2).strip()
        if dest.endswith("/") or (Path(dest).exists() and Path(dest).is_dir()):
            return [{"tool": "move_file_to_dir", "args": {"source": source, "destination_dir": dest.rstrip("/")}}]
        return [{"tool": "move_file", "args": {"source": source, "destination": dest}}]

    def copy_file(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"copy\s+([^\s]+)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        source = match.group(1).strip()
        dest = match.group(2).strip()
        if dest.endswith("/") or (Path(dest).exists() and Path(dest).is_dir()):
            return [{"tool": "copy_file_to_dir", "args": {"source": source, "destination_dir": dest.rstrip("/")}}]
        return [{"tool": "copy_file", "args": {"source": source, "destination": dest}}]

    def delete_file(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"delete\s+([^\s]+)", msg, re.IGNORECASE)
        if match:
            return [{"tool": "delete_file", "args": {"path": match.group(1).strip()}}]
        return None

    def read_file(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"(?:read|show|open)\s+([^\s]+)", msg, re.IGNORECASE)
        if match:
            return [{"tool": "read_file", "args": {"path": match.group(1).strip()}}]
        return None

    def write_file(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"write\s+(.+?)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        content = _extract_quoted(match.group(1)) or match.group(1).strip()
        path = match.group(2).strip()
        return [{"tool": "write_file", "args": {"path": path, "content": content}}]

    def append_file(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"append\s+(.+?)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        content = _extract_quoted(match.group(1)) or match.group(1).strip()
        path = match.group(2).strip()
        return [{"tool": "append_file", "args": {"path": path, "content": content}}]

    def line_count(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"(?:line\s+count|lines)\s+(?:of|in)\s+([^\s]+)", msg, re.IGNORECASE)
        if match:
            return [{"tool": "line_count", "args": {"path": match.group(1).strip()}}]
        return None

    def list_files(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        if not re.search(r"list\s+files", msg_lower):
            return None
        dir_match = re.search(r"list\s+files\s+in\s+([^\s]+)", msg, re.IGNORECASE)
        args = {"root": dir_match.group(1).strip()} if dir_match else {}
        return [{"tool": "list_files", "args": args}]

    def list_dirs(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        if not re.search(r"list\s+dirs|list\s+directories", msg_lower):
            return None
        dir_match = re.search(r"list\s+(?:dirs|directories)\s+in\s+([^\s]+)", msg, re.IGNORECASE)
        args = {"root": dir_match.group(1).strip()} if dir_match else {}
        return [{"tool": "list_dirs", "args": args}]

    def search_files(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"(?:find|search)\s+files?\s+(?:named|for)\s+(.+)", msg, re.IGNORECASE)
        if not match:
            return None
        query = _extract_quoted(match.group(1)) or match.group(1).strip()
        return [{"tool": "search_files", "args": {"query": query}}]

    def grep_files(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"(?:grep|search)\s+for\s+(.+?)\s+in\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        pattern = _extract_quoted(match.group(1)) or match.group(1).strip()
        root = match.group(2).strip()
        return [{"tool": "grep_files", "args": {"pattern": pattern, "root": root}}]

    def replace_in_file(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"replace\s+(.+?)\s+with\s+(.+?)\s+in\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        pattern = _extract_quoted(match.group(1)) or match.group(1).strip()
        replacement = _extract_quoted(match.group(2)) or match.group(2).strip()
        path = match.group(3).strip()
        return [{"tool": "replace_in_file", "args": {"path": path, "pattern": pattern, "replacement": replacement}}]

    def read_struct(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"read\s+(json|yaml|yml|ini)\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        fmt = match.group(1).lower()
        path = match.group(2).strip()
        tool = {"json": "read_json", "yaml": "read_yaml", "yml": "read_yaml", "ini": "read_ini"}[fmt]
        return [{"tool": tool, "args": {"path": path}}]

    def write_struct(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"write\s+(json|yaml|yml|ini)\s+(.+?)\s+to\s+([^\s]+)", msg, re.IGNORECASE)
        if not match:
            return None
        fmt = match.group(1).lower()
        content = _extract_quoted(match.group(2)) or match.group(2).strip()
        path = match.group(3).strip()
        tool = {"json": "write_json", "yaml": "write_yaml", "yml": "write_yaml", "ini": "write_ini"}[fmt]
        return [{"tool": tool, "args": {"path": path, "data": content}}]

    def set_struct(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(
            r"set\s+(json|yaml|yml|ini)\s+(.+?)\s+in\s+([^\s]+)\s+to\s+(.+)",
            msg,
            re.IGNORECASE,
        )
        if not match:
            return None
        fmt = match.group(1).lower()
        key_path = _extract_quoted(match.group(2)) or match.group(2).strip()
        path = match.group(3).strip()
        value = _extract_quoted(match.group(4)) or match.group(4).strip()
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

    def lint_tree(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"lint\s+(?:python\s+)?tree\s+([^\s]+)?", msg, re.IGNORECASE)
        if not match:
            return None
        root = match.group(1).strip() if match.group(1) else None
        args = {"root": root} if root else {}
        return [{"tool": "lint_python_tree", "args": args}]

    def lint_files(msg: str, msg_lower: str) -> Optional[ToolCallList]:
        match = re.search(r"lint\s+(.+)", msg, re.IGNORECASE)
        if not match:
            return None
        raw = match.group(1)
        paths = re.findall(r"[^\s]+\.py", raw)
        if not paths:
            return None
        return [{"tool": "lint_python_files", "args": {"paths": paths}}]

    patterns = [
        ToolPattern("change_intent", change_intent, priority=100),
        ToolPattern("move_file", move_file, priority=90),
        ToolPattern("copy_file", copy_file, priority=90),
        ToolPattern("delete_file", delete_file, priority=80),
        ToolPattern("read_file", read_file, priority=70),
        ToolPattern("write_file", write_file, priority=70),
        ToolPattern("append_file", append_file, priority=70),
        ToolPattern("line_count", line_count, priority=70),
        ToolPattern("list_files", list_files, priority=60),
        ToolPattern("list_dirs", list_dirs, priority=60),
        ToolPattern("search_files", search_files, priority=50),
        ToolPattern("grep_files", grep_files, priority=50),
        ToolPattern("replace_in_file", replace_in_file, priority=50),
        ToolPattern("read_struct", read_struct, priority=40),
        ToolPattern("write_struct", write_struct, priority=40),
        ToolPattern("set_struct", set_struct, priority=40),
        ToolPattern("lint_tree", lint_tree, priority=30),
        ToolPattern("lint_files", lint_files, priority=30),
    ]

    for pattern in patterns:
        detector.register(pattern)

    return detector
