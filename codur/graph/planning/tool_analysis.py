"""Tool result analysis helpers for planning."""

from langchain_core.messages import BaseMessage

from codur.graph.utils import has_tool_result, extract_list_files_output


def tool_results_include_read_file(messages: list[BaseMessage]) -> bool:
    """Check if tool results include a read_file output."""
    return has_tool_result(messages, "read_file", "read_files")


def select_file_from_tool_results(messages: list[BaseMessage]) -> str | None:
    """Select a preferred Python file from list_files tool results."""
    files = extract_list_files_output(messages)
    return pick_preferred_python_file(files)


def pick_preferred_python_file(files: list[str]) -> str | None:
    """Pick the most likely main Python file from a list."""
    if not files:
        return None
    py_files = [path for path in files if path.endswith(".py")]
    if not py_files:
        return None

    # Find all main.py or app.py files (but ignore if 3+ levels deep)
    preferred_files = [
        path for path in py_files
        if (path == "main.py" or path == "app.py" or
            path.endswith("/main.py") or path.endswith("/app.py"))
        and path.count("/") < 2
    ]

    if preferred_files:
        # Sort by: depth (fewest slashes = closest to root),
        # then prefer main.py over app.py, then shortest length
        def sort_key(p):
            is_app = p.endswith("app.py")
            return p.count("/"), is_app, len(p)
        return min(preferred_files, key=sort_key)

    # Fallback: no main.py or app.py found
    # Abort if too many candidates to choose from
    if len(py_files) > 5:
        return None
    if len(py_files) == 1:
        return py_files[0]
    return min(py_files, key=lambda p: (p.count("/"), len(p)))
