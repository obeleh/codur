"""Tool result analysis helpers for planning."""

from langchain_core.messages import BaseMessage

from codur.graph.utils import has_tool_result, extract_list_files_output


def tool_results_include_read_file(messages: list[BaseMessage]) -> bool:
    """Check if tool results include a read_file output."""
    return has_tool_result(messages, "read_file", "read_files")


def select_file_from_tool_results(messages: list[BaseMessage]) -> str | None:
    """Select a preferred Python file from list_files tool results."""
    files = extract_list_files(messages)
    return pick_preferred_python_file(files)


def extract_list_files(messages: list[BaseMessage]) -> list[str]:
    """Extract list of files from list_files tool output in messages."""
    return extract_list_files_output(messages)


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
