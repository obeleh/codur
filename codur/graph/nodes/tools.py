"""Tool execution node."""

from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from rich import console

from codur.config import CodurConfig
from codur.graph.state import AgentState, AgentStateData
from codur.graph.nodes.types import ToolNodeResult
from codur.tools import (
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
    grep_files,
    replace_in_file,
    inject_lines,
    replace_lines,
    line_count,
    read_json,
    write_json,
    set_json_value,
    read_yaml,
    write_yaml,
    set_yaml_value,
    read_ini,
    write_ini,
    set_ini_value,
    lint_python_files,
    lint_python_tree,
    list_mcp_tools,
    call_mcp_tool,
    list_mcp_resources,
    list_mcp_resource_templates,
    read_mcp_resource,
    list_tool_directory,
    get_tool_help,
    retry_in_agent,
    git_status,
    git_diff,
    git_log,
    git_stage_files,
    git_stage_all,
    git_commit,
    fetch_webpage,
    location_lookup,
    duckduckgo_search,
    convert_document,
    python_ast_graph,
    python_ast_outline,
    python_dependency_graph,
    agent_call,
)

console = console.Console()

def tool_node(state: AgentState, config: CodurConfig) -> ToolNodeResult:
    """Execute local filesystem tools requested by the planner.

    Args:
        state: Current agent state containing tool_calls list
        config: Codur configuration for MCP tool access

    Returns:
        Dictionary with agent_outcome and messages containing tool results
    """
    if "config" not in state:
        raise ValueError("AgentState must include config")
    tool_calls = state.get("tool_calls", []) or []
    root = Path.cwd()
    tool_state = state if hasattr(state, "get_config") else AgentStateData(state)
    allow_outside_root = config.runtime.allow_outside_workspace
    results = []
    errors = []
    # When adding a new tool, register it here and export it in codur/tools/__init__.py.
    last_human_msg = None
    for msg in reversed(state.get("messages", []) or []):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break
    tool_map = {
        "read_file": lambda args: read_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_file": lambda args: write_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
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
        ),
        "list_files": lambda args: list_files(root=root, state=tool_state, **args),
        "list_dirs": lambda args: list_dirs(root=root, state=tool_state, **args),
        "file_tree": lambda args: file_tree(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "search_files": lambda args: search_files(root=root, state=tool_state, **args),
        "grep_files": lambda args: grep_files(root=root, state=tool_state, **args),
        "replace_in_file": lambda args: replace_in_file(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "inject_lines": lambda args: inject_lines(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "replace_lines": lambda args: replace_lines(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "line_count": lambda args: line_count(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "read_json": lambda args: read_json(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_json": lambda args: write_json(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "set_json_value": lambda args: set_json_value(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "read_yaml": lambda args: read_yaml(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_yaml": lambda args: write_yaml(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "set_yaml_value": lambda args: set_yaml_value(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "read_ini": lambda args: read_ini(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "write_ini": lambda args: write_ini(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "set_ini_value": lambda args: set_ini_value(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "lint_python_files": lambda args: lint_python_files(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
        "lint_python_tree": lambda args: lint_python_tree(root=root, allow_outside_root=allow_outside_root, state=tool_state, **args),
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
        "python_dependency_graph": lambda args: python_dependency_graph(
            root=root,
            allow_outside_root=allow_outside_root,
            state=tool_state,
            **args,
        ),
        "agent_call": lambda args: agent_call(
            config=config,
            state=tool_state,
            **args,
        ),
    }

    for i, call in enumerate(tool_calls):
        if config.verbose:
            console.log(f"Executing tool call: {call}")
        tool_name = call.get("tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        _normalize_tool_args(args)

        # For agent_call, inject file_contents from previous read_file result
        if tool_name == "agent_call" and i > 0:
            prev_call = tool_calls[i - 1]
            if prev_call.get("tool") == "read_file" and results:
                prev_result = results[-1]
                if prev_result.get("tool") == "read_file":
                    args["file_contents"] = prev_result.get("output", "")

        if tool_name not in tool_map:
            errors.append(f"Unknown tool: {tool_name}")
            continue
        try:
            output = tool_map[tool_name](args)
            results.append({"tool": tool_name, "output": output})
        except Exception as exc:
            errors.append(f"{tool_name} failed: {exc}")

    summary_lines = []
    for item in results:
        summary_lines.append(f"{item['tool']}: {item['output']}")
    for error in errors:
        summary_lines.append(f"error: {error}")
    summary = "\n".join(summary_lines) if summary_lines else "No tool calls executed."

    return {
        "agent_outcome": {
            "agent": "tools",
            "result": summary,
            "status": "success" if not errors else "error",
        },
        "messages": [SystemMessage(content=f"Tool results:\n{summary}")],
        "llm_calls": tool_state.get("llm_calls", state.get("llm_calls", 0)),
    }


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
