"""Shared tool execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

from langchain_core.messages import HumanMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState, AgentStateData
from codur.graph.state_operations import get_messages
from codur.utils.path_utils import resolve_path
from codur.tools.filesystem import (
    read_file,
    write_file,
    append_file,
    delete_file,
    copy_file,
    move_file,
    copy_file_to_dir,
    move_file_to_dir,
    list_files,
    list_dirs,
    file_tree,
    search_files,
    replace_in_file,
    inject_lines,
    replace_lines,
    line_count,
)
from codur.tools.ripgrep import (
    grep_files,
    ripgrep_search,
)
from codur.tools.structured import (
    read_json,
    write_json,
    set_json_value,
    read_yaml,
    write_yaml,
    set_yaml_value,
    read_ini,
    write_ini,
    set_ini_value,
)
from codur.tools.linting import (
    lint_python_files,
    lint_python_tree,
)
from codur.tools.mcp_tools import (
    list_mcp_tools,
    call_mcp_tool,
    list_mcp_resources,
    list_mcp_resource_templates,
    read_mcp_resource,
)
from codur.tools.registry import (
    list_tool_directory,
    get_tool_help,
)
from codur.tools.git import (
    git_status,
    git_diff,
    git_log,
    git_stage_files,
    git_stage_all,
    git_commit,
)
from codur.tools.webrequests import (
    fetch_webpage,
    location_lookup,
)
from codur.tools.duckduckgo import (
    duckduckgo_search,
)
from codur.tools.pandoc import (
    convert_document,
)
from codur.tools.python_ast import (
    python_ast_graph,
    python_ast_outline,
    python_ast_dependencies,
    python_ast_dependencies_multifile,
)
from codur.tools.project_analysis import (
    python_dependency_graph,
    code_quality,
)
from codur.tools.psutil_tools import (
    system_cpu_stats,
    system_memory_stats,
    system_disk_usage,
    system_process_snapshot,
    system_processes_top,
    system_processes_list,
)
from codur.tools.code_modification import (
    inject_function,
    replace_function,
    replace_class,
    replace_method,
    replace_file_content,
)
from codur.tools.rope_tools import (
    rope_find_usages,
    rope_find_definition,
    rope_rename_symbol,
    rope_move_module,
    rope_extract_method,
)

console = Console()

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


def execute_tool_calls(
    tool_calls: list[dict],
    state: AgentState,
    config: CodurConfig,
    *,
    augment: bool = True,
    summary_mode: str = "full",
) -> ToolExecutionResult:
    """Execute tool calls using a shared tool map."""
    root = Path.cwd()
    allow_outside_root = config.runtime.allow_outside_workspace
    tool_state = state if hasattr(state, "get_config") else AgentStateData(state)

    tool_calls = list(tool_calls or [])
    if augment:
        tool_calls = _augment_tool_calls(tool_calls)

    results = []
    errors = []
    last_read_file_output = None
    has_multifile_call = any(call.get("tool") == "python_ast_dependencies_multifile" for call in tool_calls)

    last_human_msg = _last_human_message(get_messages(state) or [])
    tool_map = _build_tool_map(root, allow_outside_root, tool_state, last_human_msg, config)

    i = 0
    while i < len(tool_calls):
        call = tool_calls[i]
        if config.verbose:
            console.log(f"Executing tool call: {call}")
        tool_name = call.get("tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        _normalize_tool_args(args)

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
    return ToolExecutionResult(results=results, errors=errors, summary=summary)


def get_tool_names(state: AgentState, config: CodurConfig) -> set[str]:
    """Return the set of supported tool names."""
    root = Path.cwd()
    allow_outside_root = config.runtime.allow_outside_workspace
    tool_state = state if hasattr(state, "get_config") else AgentStateData(state)
    last_human_msg = _last_human_message(get_messages(state) or [])
    tool_map = _build_tool_map(root, allow_outside_root, tool_state, last_human_msg, config)
    return set(tool_map.keys())


def _build_tool_map(
    root: Path,
    allow_outside_root: bool,
    tool_state: AgentState,
    last_human_msg: Optional[str],
    config: CodurConfig,
) -> dict:
    from codur.tools.agents import agent_call, retry_in_agent

    def _guarded_write_file(args: dict) -> str:
        _guard_test_file_overwrite(
            args.get("path"),
            root,
            allow_outside_root,
            last_human_msg,
        )
        return write_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args)

    def _guarded_replace_file_content(args: dict) -> str:
        _guard_test_file_overwrite(
            args.get("path"),
            root,
            allow_outside_root,
            last_human_msg,
        )
        return replace_file_content(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args)

    return {
        "read_file": lambda args: read_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_file": _guarded_write_file,
        "append_file": lambda args: append_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "delete_file": lambda args: delete_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "copy_file": lambda args: copy_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "move_file": lambda args: move_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "copy_file_to_dir": lambda args: copy_file_to_dir(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "move_file_to_dir": lambda args: move_file_to_dir(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "retry_in_agent": lambda args: retry_in_agent(
            task=args.get("task") or (last_human_msg or ""),
            agent=args.get("agent", ""),
            state=tool_state,
            config=config,
            reason=args.get("reason"),
        ),
        "list_files": lambda args: list_files(root=root, state=tool_state, **args),
        "list_dirs": lambda args: list_dirs(root=root, state=tool_state, **args),
        "file_tree": lambda args: file_tree(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "search_files": lambda args: search_files(root=root, state=tool_state, **args),
        "grep_files": lambda args: grep_files(root=root, state=tool_state, **args),
        "ripgrep_search": lambda args: ripgrep_search(root=root, state=tool_state, **args),
        "replace_in_file": lambda args: replace_in_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "inject_lines": lambda args: inject_lines(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "replace_lines": lambda args: replace_lines(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "line_count": lambda args: line_count(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "inject_function": lambda args: inject_function(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "replace_function": lambda args: replace_function(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "replace_class": lambda args: replace_class(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "replace_method": lambda args: replace_method(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "replace_file_content": _guarded_replace_file_content,
        "read_json": lambda args: read_json(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_json": lambda args: write_json(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "set_json_value": lambda args: set_json_value(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "read_yaml": lambda args: read_yaml(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_yaml": lambda args: write_yaml(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "set_yaml_value": lambda args: set_yaml_value(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "read_ini": lambda args: read_ini(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_ini": lambda args: write_ini(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "set_ini_value": lambda args: set_ini_value(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "lint_python_files": lambda args: lint_python_files(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "lint_python_tree": lambda args: lint_python_tree(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "list_mcp_tools": lambda args: list_mcp_tools(state=tool_state, **args),
        "call_mcp_tool": lambda args: call_mcp_tool(state=tool_state, **args),
        "list_mcp_resources": lambda args: list_mcp_resources(state=tool_state, **args),
        "list_mcp_resource_templates": lambda args: list_mcp_resource_templates(state=tool_state, **args),
        "read_mcp_resource": lambda args: read_mcp_resource(state=tool_state, **args),
        "list_tool_directory": lambda args: list_tool_directory(state=tool_state, **args),
        "get_tool_help": lambda args: get_tool_help(state=tool_state, **args),
        "git_status": lambda args: git_status(root=root, state=tool_state, **args),
        "git_diff": lambda args: git_diff(root=root, state=tool_state, **args),
        "git_log": lambda args: git_log(root=root, state=tool_state, **args),
        "git_stage_files": lambda args: git_stage_files(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "git_stage_all": lambda args: git_stage_all(root=root, state=tool_state, **args),
        "git_commit": lambda args: git_commit(root=root, state=tool_state, **args),
        "fetch_webpage": lambda args: fetch_webpage(state=tool_state, **args),
        "location_lookup": lambda args: location_lookup(state=tool_state, **args),
        "duckduckgo_search": lambda args: duckduckgo_search(state=tool_state, **args),
        "convert_document": lambda args: convert_document(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "python_ast_graph": lambda args: python_ast_graph(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "python_ast_outline": lambda args: python_ast_outline(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "python_ast_dependencies": lambda args: python_ast_dependencies(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "python_ast_dependencies_multifile": lambda args: python_ast_dependencies_multifile(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "python_dependency_graph": lambda args: python_dependency_graph(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "code_quality": lambda args: code_quality(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "rope_find_usages": lambda args: rope_find_usages(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "rope_find_definition": lambda args: rope_find_definition(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "rope_rename_symbol": lambda args: rope_rename_symbol(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "rope_move_module": lambda args: rope_move_module(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "rope_extract_method": lambda args: rope_extract_method(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "system_cpu_stats": lambda args: system_cpu_stats(state=tool_state, **args),
        "system_memory_stats": lambda args: system_memory_stats(state=tool_state, **args),
        "system_disk_usage": lambda args: system_disk_usage(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "system_process_snapshot": lambda args: system_process_snapshot(state=tool_state, **args),
        "system_processes_top": lambda args: system_processes_top(state=tool_state, **args),
        "system_processes_list": lambda args: system_processes_list(state=tool_state, **args),
        "agent_call": lambda args: agent_call(
            config=config,
            state=tool_state,
            **args,
        ),
    }


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
    """Add python_ast_dependencies when reading Python files."""
    augmented: list[dict] = []
    existing_deps: set[str] = set()
    for call in tool_calls:
        if call.get("tool") == "python_ast_dependencies":
            args = call.get("args", {})
            if isinstance(args, dict):
                path = args.get("path")
                if isinstance(path, str):
                    existing_deps.add(_strip_at_prefix(path))

    for call in tool_calls:
        augmented.append(call)
        if call.get("tool") != "read_file":
            continue
        args = call.get("args", {})
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str):
            continue
        clean_path = _strip_at_prefix(path)
        if not clean_path.endswith(".py"):
            continue
        if clean_path in existing_deps:
            continue
        augmented.append({"tool": "python_ast_dependencies", "args": {"path": clean_path}})
        existing_deps.add(clean_path)
    return augmented


def _normalize_tool_args(args: dict) -> None:
    """Normalize tool arguments in-place for common shorthand formats."""
    for key, value in list(args.items()):
        if isinstance(value, str):
            args[key] = _strip_at_prefix(value)
        elif isinstance(value, list):
            args[key] = [_strip_at_prefix(item) if isinstance(item, str) else item for item in value]


def _strip_at_prefix(value: str) -> str:
    if value.startswith("@") and len(value) > 1:
        return value[1:]
    return value


def _last_human_message(messages: list) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None
