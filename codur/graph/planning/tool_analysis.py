"""Tool result analysis helpers for planning."""

import ast
from langchain_core.messages import SystemMessage, BaseMessage


def tool_results_include_read_file(messages: list[BaseMessage]) -> bool:
    """Check if tool results include a read_file output."""
    for msg in messages:
        if isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:"):
            if "read_file:" in msg.content:
                return True
    return False


def select_file_from_tool_results(messages: list[BaseMessage]) -> str | None:
    """Select a preferred Python file from list_files tool results."""
    files = extract_list_files(messages)
    return pick_preferred_python_file(files)


def extract_list_files(messages: list[BaseMessage]) -> list[str]:
    """Extract list of files from list_files tool output in messages."""
    for msg in messages:
        if not isinstance(msg, SystemMessage):
            continue
        if not msg.content.startswith("Tool results:"):
            continue
        for line in msg.content.splitlines():
            if not line.startswith("list_files:"):
                continue
            payload = line.split("list_files:", 1)[1].strip()
            try:
                parsed = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                return []
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, str)]
    return []


def pick_preferred_python_file(files: list[str]) -> str | None:
    """Pick the most likely main Python file from a list."""
    if not files:
        return None
    py_files = [path for path in files if path.endswith(".py")]
    if not py_files:
        return None
    for preferred in ("app.py", "main.py"):
        if preferred in py_files:
            return preferred
    for preferred in ("app.py", "main.py"):
        matches = [path for path in py_files if path.endswith(f"/{preferred}")]
        if matches:
            return min(matches, key=lambda p: (p.count("/"), len(p)))
    if len(py_files) == 1:
        return py_files[0]
    return min(py_files, key=lambda p: (p.count("/"), len(p)))
