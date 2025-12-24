"""
Tool registry for Codur.
"""

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
    grep_files,
    replace_in_file,
    inject_lines,
    replace_lines,
    line_count,
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
from codur.tools.agents import (
    retry_in_agent,
    agent_call,
)
from codur.tools.mcp_tools import (
    list_mcp_tools,
    call_mcp_tool,
    list_mcp_resources,
    list_mcp_resource_templates,
    read_mcp_resource,
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
from codur.tools.validation import (
    validate_python_syntax,
)
from codur.tools.ast_utils import (
    find_function_lines,
    find_class_lines,
    find_method_lines,
)
from codur.tools.code_modification import (
    inject_function,
)
from codur.tools.registry import (
    list_tool_directory,
    get_tool_help,
)
from codur.tools.psutil_tools import (
    system_cpu_stats,
    system_memory_stats,
    system_disk_usage,
    system_process_snapshot,
    system_processes_top,
    system_processes_list,
)

__all__ = [
    "read_file",
    "write_file",
    "append_file",
    "delete_file",
    "copy_file",
    "move_file",
    "copy_file_to_dir",
    "move_file_to_dir",
    "list_files",
    "list_dirs",
    "file_tree",
    "search_files",
    "grep_files",
    "replace_in_file",
    "inject_lines",
    "replace_lines",
    "line_count",
    "read_json",
    "write_json",
    "set_json_value",
    "read_yaml",
    "write_yaml",
    "set_yaml_value",
    "read_ini",
    "write_ini",
    "set_ini_value",
    "lint_python_files",
    "lint_python_tree",
    "retry_in_agent",
    "agent_call",
    "list_mcp_tools",
    "call_mcp_tool",
    "list_mcp_resources",
    "list_mcp_resource_templates",
    "read_mcp_resource",
    "git_status",
    "git_diff",
    "git_log",
    "git_stage_files",
    "git_stage_all",
    "git_commit",
    "fetch_webpage",
    "location_lookup",
    "duckduckgo_search",
    "convert_document",
    "python_ast_graph",
    "python_ast_outline",
    "python_ast_dependencies",
    "python_ast_dependencies_multifile",
    "python_dependency_graph",
    "code_quality",
    "validate_python_syntax",
    "find_function_lines",
    "find_class_lines",
    "find_method_lines",
    "inject_function",
    "list_tool_directory",
    "get_tool_help",
    "system_cpu_stats",
    "system_memory_stats",
    "system_disk_usage",
    "system_process_snapshot",
    "system_processes_top",
    "system_processes_list",
]
