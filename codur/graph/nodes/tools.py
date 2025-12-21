"""Tool execution node."""

from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

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
)


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
    }

    for call in tool_calls:
        tool_name = call.get("tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
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
    }
