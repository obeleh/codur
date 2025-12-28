"""Shared tool execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Callable, List
import re

from langchain_core.messages import HumanMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState, AgentStateData
from codur.graph.state_operations import get_messages, is_verbose
from codur.utils.path_utils import resolve_path

console = Console()


def _format_syntax_validation_result(result: tuple[bool, Optional[str]]) -> str:
    """Format the result of validate_python_syntax for tool output."""
    is_valid, error_msg = result
    if is_valid:
        return "✓ Python syntax is valid"
    else:
        return f"✗ Syntax error:\n{error_msg}"


_TEST_OVERWRITE_VERBS = {
    "overwrite",
    "replace",
    "rewrite",
    "regenerate",
    "recreate",
    "reset",
}
_TEST_WRITE_VERBS = {
    "write",
    "add",
    "update",
    "create",
    "implement",
    "generate",
}


def _is_test_path(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    parts = {part.lower() for part in path.parts}
    return "tests" in parts or "test" in parts


def _allows_test_overwrite(last_human_msg: Optional[str], path: Path) -> bool:
    if not last_human_msg:
        return False
    msg_lower = last_human_msg.lower()
    filename = path.name.lower()
    has_test_reference = (
        (filename and filename in msg_lower)
        or "test file" in msg_lower
        or "unit test" in msg_lower
        or re.search(r"\btests?\b", msg_lower) is not None
    )
    if any(verb in msg_lower for verb in _TEST_OVERWRITE_VERBS):
        return has_test_reference
    if any(verb in msg_lower for verb in _TEST_WRITE_VERBS):
        return has_test_reference or ("unit test" in msg_lower or "tests" in msg_lower)
    return False


def _guard_test_file_overwrite(
    path: Optional[str],
    root: Path,
    allow_outside_root: bool,
    last_human_msg: Optional[str],
) -> None:
    if not path:
        return
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    if not target.exists():
        return
    if not _is_test_path(target):
        return
    if _allows_test_overwrite(last_human_msg, target):
        return
    raise ValueError(
        f"Refusing to overwrite existing test file '{path}' without explicit request "
        "to replace/overwrite it."
    )


@dataclass
class ToolExecutionResult:
    results: list[dict]
    errors: list[str]
    summary: str
    error_details: Optional[str] = None


def _inject_missing_required_params(tool_calls: list[dict], state: AgentState) -> None:
    """Inject missing required parameters when they can be inferred from context.

    Shows warnings when parameters are injected, indicating the LLM omitted them.
    """
    CODE_TOOLS_REQUIRING_PATH = {
        "replace_function", "replace_class", "replace_method",
        "replace_file_content", "inject_function", "write_file",
        "read_file", "append_file", "delete_file",
    }

    # Try to infer path from recent tool calls
    from codur.graph.state_operations import get_tool_calls
    try:
        previous_calls = get_tool_calls(state)
    except Exception:
        previous_calls = []

    inferred_path = None
    for call in previous_calls:
        args = call.get("args", {})
        path = args.get("path") or args.get("file_path")
        if path:
            inferred_path = path
            break

    # Check and inject missing paths
    for call in tool_calls:
        tool_name = call.get("tool")
        args = call.get("args", {})

        if tool_name in CODE_TOOLS_REQUIRING_PATH and "path" not in args:
            if inferred_path:
                console.log(f"[yellow]⚠ Warning: Injecting missing 'path' for {tool_name}: {inferred_path}[/yellow]")
                console.log(f"[dim]  LLM omitted required parameter - possible context/token limit issue[/dim]")
                args["path"] = inferred_path
            else:
                console.log(f"[red]✗ {tool_name} missing required 'path' and none could be inferred[/red]")


def execute_tool_calls(
    tool_calls: list[dict],
    state: AgentState,
    config: CodurConfig,
    *,
    augment: bool = True,
    summary_mode: str = "full", # brief|full
) -> ToolExecutionResult:
    if is_verbose(state):
        console.log(f"[cyan]Executing {len(tool_calls)} tool call(s)...[/cyan]")

    """Execute tool calls using a shared tool map."""
    root = Path.cwd()
    allow_outside_root = config.runtime.allow_outside_workspace
    tool_state = state if hasattr(state, "get_config") else AgentStateData(state)

    tool_calls = list(tool_calls or [])
    if augment:
        tool_calls = _augment_tool_calls(tool_calls)

    # Inject missing required parameters when possible (with warnings)
    _inject_missing_required_params(tool_calls, state)

    results = []
    errors = []
    error_details = []
    last_read_file_output = None
    has_multifile_call = any(call.get("tool") == "python_ast_dependencies_multifile" for call in tool_calls)

    last_human_msg = _last_human_message(get_messages(state) or [])
    tool_map = _build_tool_map(root, allow_outside_root, tool_state, last_human_msg, config)

    i = 0
    while i < len(tool_calls):
        call = tool_calls[i]
        tool_name = call.get("tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        _normalize_tool_args(args)
        if config.verbose:
            console.log(f"Executing tool call: {_format_tool_call_for_log(tool_name, args)}")

        # For agent_call, inject file_contents from the most recent read_file result
        if tool_name == "agent_call" and last_read_file_output is not None:
            args["file_contents"] = last_read_file_output

        if tool_name not in tool_map:
            errors.append(f"Unknown tool: {tool_name}")
            i += 1
            continue
        try:
            output = tool_map[tool_name](args)
            results.append({"tool": tool_name, "output": output, "args": dict(args)})
            if tool_name == "read_file":
                last_read_file_output = output
            if tool_name == "list_files" and not has_multifile_call:
                py_files = []
                if isinstance(output, list):
                    py_files = [item for item in output if isinstance(item, str) and item.endswith(".py")]
                if 0 < len(py_files) <= 5:
                    tool_calls.insert(
                        i + 1,
                        {"tool": "python_ast_dependencies_multifile", "args": {"paths": py_files}},
                    )
                    has_multifile_call = True
        except Exception as exc:
            errors.append(f"{tool_name} failed: {exc}")
        i += 1

    summary = _format_summary(results, errors, summary_mode)
    return ToolExecutionResult(
        results=results,
        errors=errors,
        summary=summary,
        error_details="\n".join(error_details) if error_details else None,
    )


def get_tool_names(state: AgentState, config: CodurConfig) -> set[str]:
    """Return the set of supported tool names."""
    root = Path.cwd()
    allow_outside_root = config.runtime.allow_outside_workspace
    tool_state = state if hasattr(state, "get_config") else AgentStateData(state)
    last_human_msg = _last_human_message(get_messages(state) or [])
    tool_map = _build_tool_map(root, allow_outside_root, tool_state, last_human_msg, config)
    return set(tool_map.keys())


# Tool metadata for dynamic wiring
_UTILITY_TOOLS = {
    "find_function_lines", "find_class_lines", "find_method_lines",
    "markdown_outline", "markdown_extract_sections", "markdown_extract_tables",
    "discover_entry_points", "get_primary_entry_point",
}

_GUARDED_TOOLS = {"write_file", "replace_file_content"}
_OUTPUT_FORMATTERS = {"validate_python_syntax": _format_syntax_validation_result}
_CONFIG_TOOLS = {"agent_call", "retry_in_agent"}

# Tools that need root + allow_outside_root context
_FILESYSTEM_CONTEXT_TOOLS = {
    "read_file", "write_file", "append_file", "delete_file",
    "copy_file", "move_file", "copy_file_to_dir", "move_file_to_dir",
    "file_tree", "replace_in_file", "inject_lines", "replace_lines",
    "line_count", "inject_function", "replace_function", "replace_class",
    "replace_method", "replace_file_content", "read_json", "write_json",
    "set_json_value", "read_yaml", "write_yaml", "set_yaml_value",
    "read_ini", "write_ini", "set_ini_value", "lint_python_files",
    "lint_python_tree", "system_disk_usage", "git_stage_files",
    "convert_document", "python_ast_graph", "python_ast_outline",
    "python_ast_dependencies", "python_ast_dependencies_multifile",
    "python_dependency_graph", "code_quality", "rope_find_usages",
    "rope_find_definition", "rope_rename_symbol", "rope_move_module",
    "rope_extract_method"
}

# Tools that need search_files context
_SEARCH_CONTEXT_TOOLS = {"search_files", "grep_files", "list_files", "list_dirs"}


def _build_tool_map(
    root: Path,
    allow_outside_root: bool,
    tool_state: AgentState,
    last_human_msg: Optional[str],
    config: CodurConfig,
) -> dict:
    """Build tool map dynamically from tool registry."""
    from codur.tools import __all__ as tool_names
    tool_map = {}

    # Import tools dynamically
    import codur.tools as tools_module

    for tool_name in tool_names:
        # Skip utility functions that aren't meant for direct LLM invocation
        if tool_name in _UTILITY_TOOLS:
            continue

        # Get the tool function
        tool_func = getattr(tools_module, tool_name, None)
        if not callable(tool_func):
            continue

        # Handle special cases
        if tool_name in _GUARDED_TOOLS:
            # Create guarded wrapper for write_file and replace_file_content
            if tool_name == "write_file":
                def _guarded_write_file(args: dict, tf=tool_func) -> str:
                    _guard_test_file_overwrite(
                        args.get("path"),
                        root,
                        allow_outside_root,
                        last_human_msg,
                    )
                    return tf(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args)
                tool_map[tool_name] = _guarded_write_file
            elif tool_name == "replace_file_content":
                def _guarded_replace_file_content(args: dict, tf=tool_func) -> str:
                    _guard_test_file_overwrite(
                        args.get("path"),
                        root,
                        allow_outside_root,
                        last_human_msg,
                    )
                    return tf(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args)
                tool_map[tool_name] = _guarded_replace_file_content

        elif tool_name in _OUTPUT_FORMATTERS:
            # Format output for specific tools
            formatter = _OUTPUT_FORMATTERS[tool_name]
            tool_map[tool_name] = lambda args, tf=tool_func, fmt=formatter: fmt(tf(args.get("code", "")))

        elif tool_name in _CONFIG_TOOLS:
            # Tools that need config parameter
            if tool_name == "agent_call":
                tool_map[tool_name] = lambda args, tf=tool_func: tf(config=config, state=tool_state, **args)
            elif tool_name == "retry_in_agent":
                tool_map[tool_name] = lambda args, tf=tool_func: tf(
                    task=args.get("task") or (last_human_msg or ""),
                    agent=args.get("agent", ""),
                    state=tool_state,
                    config=config,
                    reason=args.get("reason"),
                )

        elif tool_name in _FILESYSTEM_CONTEXT_TOOLS:
            # Tools that need filesystem context
            tool_map[tool_name] = lambda args, tf=tool_func: tf(
                root=root,
                allow_outside_root=allow_outside_root,
                state=tool_state,
                **args
            )

        elif tool_name in _SEARCH_CONTEXT_TOOLS:
            # Tools that need search context
            tool_map[tool_name] = lambda args, tf=tool_func: tf(root=root, state=tool_state, **args)

        else:
            # Default: just inject state
            tool_map[tool_name] = lambda args, tf=tool_func: tf(state=tool_state, **args)

    return tool_map


def _format_summary(results: list[dict], errors: list[str], mode: str) -> str:
    summary_lines = []
    for item in results:
        summary_lines.append(_format_tool_result(item, mode))
    for error in errors:
        summary_lines.append(f"error: {error}")
    return "\n".join(summary_lines) if summary_lines else "No tool calls executed."


def _format_tool_result(item: dict, mode: str) -> str:
    tool = item.get("tool")
    output = item.get("output")
    args = item.get("args", {})
    if mode == "brief" and tool == "read_file":
        path = args.get("path", "unknown")
        length = len(output) if isinstance(output, str) else len(str(output))
        return f"read_file: {path} -> {length} chars"
    if mode == "brief" and tool == "write_file":
        path = args.get("path", "unknown")
        return f"write_file: {path} -> {output}"
    return f"{tool}: {output}"


def _augment_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Add python_ast_dependencies when reading Python files and validate_python_syntax after code modifications."""
    augmented: list[dict] = []
    existing_deps: set[str] = set()
    for call in tool_calls:
        if call.get("tool") == "python_ast_dependencies":
            args = call.get("args", {})
            if isinstance(args, dict):
                path = args.get("path")
                if isinstance(path, str):
                    existing_deps.add(_strip_at_prefix(path))

    # Code modification tools that operate on Python files
    python_code_modification_tools = {
        "write_file", "replace_function", "replace_class",
        "replace_method", "replace_file_content", "inject_function"
    }

    for call in tool_calls:
        augmented.append(call)
        tool = call.get("tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            continue

        path = args.get("path")
        if not isinstance(path, str):
            continue
        clean_path = _strip_at_prefix(path)

        # Add dependency analysis after reading Python files
        if tool == "read_file":
            if not clean_path.endswith(".py"):
                continue
            if clean_path in existing_deps:
                continue
            augmented.append({"tool": "python_ast_dependencies", "args": {"path": clean_path}})
            existing_deps.add(clean_path)

        # Add syntax validation after modifying Python files
        elif tool in python_code_modification_tools:
            if not clean_path.endswith(".py"):
                continue
            # Extract the code to validate - use new_code, content, or code field
            code_to_validate = args.get("new_code") or args.get("content") or args.get("code", "")
            if code_to_validate:
                augmented.append({"tool": "validate_python_syntax", "args": {"code": code_to_validate}})

    return augmented


def _normalize_tool_args(args: dict) -> None:
    """Normalize tool arguments in-place for common shorthand formats."""
    for key, value in list(args.items()):
        if isinstance(value, str):
            args[key] = _strip_at_prefix(value)
        elif isinstance(value, list):
            args[key] = [_strip_at_prefix(item) if isinstance(item, str) else item for item in value]


def _format_tool_call_for_log(tool_name: Optional[str], args: dict) -> dict:
    """Render a log-friendly tool call dict without internal fields."""
    return {"tool": tool_name, "args": dict(args) if isinstance(args, dict) else {}}


def _strip_at_prefix(value: str) -> str:
    if value.startswith("@") and len(value) > 1:
        return value[1:]
    return value


def _last_human_message(messages: list) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None
