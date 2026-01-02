"""
Tool registry for Codur.
"""

from codur.tools.filesystem import (
    read_file,
    read_files,
    write_file,
    write_files,
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
    json_decode,
    write_json,
    set_json_value,
    read_yaml,
    yaml_decode,
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
from codur.tools.markdown import (
    markdown_outline,
    markdown_extract_sections,
    markdown_extract_tables,
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
    run_python_file,
    run_pytest,
)
from codur.tools.meta_tools import (
    build_verification_response,
    clarify,
    done,
)
from codur.tools.ast_utils import (
    find_function_lines,
    find_class_lines,
    find_method_lines,
)
from codur.tools.code_modification import (
    replace_function,
    replace_class,
    replace_method,
    replace_file_content,
    inject_function,
)
from codur.tools.rope_tools import (
    rope_find_usages,
    rope_find_definition,
    rope_rename_symbol,
    rope_move_module,
    rope_extract_method,
)
from codur.tools.registry import (
    list_tool_directory,
    get_tool_help,
    list_tools_for_tasks,
)
from codur.tools.psutil_tools import (
    system_cpu_stats,
    system_memory_stats,
    system_disk_usage,
    system_process_snapshot,
    system_processes_top,
    system_processes_list,
)
from codur.tools.project_discovery import (
    discover_entry_points,
    get_primary_entry_point,
)

__all__ = [
    "read_file",
    "read_files",
    "write_file",
    "write_files",
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
    "ripgrep_search",
    "replace_in_file",
    "inject_lines",
    "replace_lines",
    "line_count",
    "read_json",
    "json_decode",
    "write_json",
    "set_json_value",
    "read_yaml",
    "yaml_decode",
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
    "markdown_outline",
    "markdown_extract_sections",
    "markdown_extract_tables",
    "python_ast_graph",
    "python_ast_outline",
    "python_ast_dependencies",
    "python_ast_dependencies_multifile",
    "python_dependency_graph",
    "code_quality",
    "validate_python_syntax",
    "run_python_file",
    "run_pytest",
    "build_verification_response",
    "find_function_lines",
    "find_class_lines",
    "find_method_lines",
    "replace_function",
    "replace_class",
    "replace_method",
    "replace_file_content",
    "inject_function",
    "rope_find_usages",
    "rope_find_definition",
    "rope_rename_symbol",
    "rope_move_module",
    "rope_extract_method",
    "list_tool_directory",
    "get_tool_help",
    "list_tools_for_tasks",
    "system_cpu_stats",
    "system_memory_stats",
    "system_disk_usage",
    "system_process_snapshot",
    "system_processes_top",
    "system_processes_list",
    "discover_entry_points",
    "get_primary_entry_point",
    "clarify",
    "done",
]
