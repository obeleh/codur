"""
High-level code modification tools for the coding agent.
These tools combine AST analysis with file modification to robustly update code entities.
"""

from pathlib import Path
from typing import Optional, Any
import ast
import re
import textwrap

from codur.constants import TaskType
from codur.tools.filesystem import read_file, replace_lines, write_file, inject_lines
from codur.tools.ast_utils import find_function_lines, find_class_lines, find_method_lines
from codur.tools.tool_annotations import (
    ToolContext,
    ToolGuard,
    ToolSideEffect,
    tool_contexts,
    tool_guards,
    tool_scenarios,
    tool_side_effects,
)
from codur.tools.validation import validate_python_syntax


def _validate_with_dedent(code: str) -> tuple[bool, Optional[str]]:
    """Validate syntax, attempting dedent if initial parse fails due to indentation."""
    is_valid, error_msg = validate_python_syntax(code)
    if not is_valid and "unexpected indent" in str(error_msg):
        # Try dedenting
        dedented_code = textwrap.dedent(code)
        is_valid_dedent, error_msg_dedent = validate_python_syntax(dedented_code)
        if is_valid_dedent:
            return True, None
    return is_valid, error_msg


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.COMPLEX_REFACTOR)
def replace_function(
    path: str,
    function_name: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> str:
    """Replace a specific function implementation in a file.

    Args:
        path: Path to the python file
        function_name: Name of the function to replace
        new_code: Complete new source code for the function
        root: Project root directory (defaults to cwd)
        allow_outside_root: Whether to allow writing outside root
        state: Agent state (optional)

    Returns:
        Success message or error description
    """
    if root is None:
        root = Path.cwd()

    # Validate syntax first
    is_valid, error_msg = _validate_with_dedent(new_code)
    if not is_valid:
        # TODO: Create fallback mechanism that can work with invalid code. Maybe LSP?
        return f"Invalid Python syntax in replacement code:\n{error_msg}\n\nCode attempted:\n{new_code}"

    try:
        # Read current content to find lines
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        
        line_range = find_function_lines(content, function_name)
        if not line_range:
            return f"Could not find function '{function_name}' in {path}"
        
        start_line, end_line = line_range
        
        # Apply replacement
        replace_lines(
            path=path,
            start_line=start_line,
            end_line=end_line,
            content=new_code,
            root=root,
            allow_outside_root=allow_outside_root,
            state=state
        )
        
        return f"Successfully replaced function '{function_name}' in {path}"

    except Exception as e:
        return f"Failed to replace function: {str(e)}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.COMPLEX_REFACTOR)
def replace_class(
    path: str,
    class_name: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> str:
    """Replace a specific class implementation in a file.

    Args:
        path: Path to the python file
        class_name: Name of the class to replace
        new_code: Complete new source code for the class
        root: Project root directory (defaults to cwd)
        allow_outside_root: Whether to allow writing outside root
        state: Agent state (optional)

    Returns:
        Success message or error description
    """
    if root is None:
        root = Path.cwd()

    is_valid, error_msg = _validate_with_dedent(new_code)
    if not is_valid:
        return f"Invalid Python syntax in replacement code:\n{error_msg}\n\nCode attempted:\n{new_code}"

    try:
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        
        line_range = find_class_lines(content, class_name)
        if not line_range:
            return f"Could not find class '{class_name}' in {path}"
        
        start_line, end_line = line_range
        
        replace_lines(
            path=path,
            start_line=start_line,
            end_line=end_line,
            content=new_code,
            root=root,
            allow_outside_root=allow_outside_root,
            state=state
        )
        
        return f"Successfully replaced class '{class_name}' in {path}"

    except Exception as e:
        return f"Failed to replace class: {str(e)}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.COMPLEX_REFACTOR)
def replace_method(
    path: str,
    class_name: str,
    method_name: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> str:
    """Replace a specific method implementation in a class.

    Args:
        path: Path to the python file
        class_name: Name of the class containing the method
        method_name: Name of the method to replace
        new_code: Complete new source code for the method
        root: Project root directory (defaults to cwd)
        allow_outside_root: Whether to allow writing outside root
        state: Agent state (optional)

    Returns:
        Success message or error description
    """
    if root is None:
        root = Path.cwd()

    is_valid, error_msg = _validate_with_dedent(new_code)
    if not is_valid:
        return f"Invalid Python syntax in replacement code:\n{error_msg}\n\nCode attempted:\n{new_code}"

    try:
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        
        line_range = find_method_lines(content, class_name, method_name)
        if not line_range:
            return f"Could not find method '{class_name}.{method_name}' in {path}"
        
        start_line, end_line = line_range
        
        replace_lines(
            path=path,
            start_line=start_line,
            end_line=end_line,
            content=new_code,
            root=root,
            allow_outside_root=allow_outside_root,
            state=state
        )
        
        return f"Successfully replaced method '{class_name}.{method_name}' in {path}"

    except Exception as e:
        return f"Failed to replace method: {str(e)}"


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_guards(ToolGuard.TEST_OVERWRITE)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.COMPLEX_REFACTOR)
def replace_file_content(
    path: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> str:
    """Replace the entire content of a file (alias for write_file with syntax validation).

    Args:
        path: Path to the file
        new_code: New content
        root: Project root directory
        allow_outside_root: Whether to allow writing outside root
        state: Agent state

    Returns:
        Success message or error description
    """
    if root is None:
        root = Path.cwd()

    if path.endswith(".py"):
        is_valid, error_msg = validate_python_syntax(new_code)
        if not is_valid:
            return f"Invalid Python syntax:\n{error_msg}\n\nCode attempted:\n{new_code}"

    try:
        write_file(
            path=path,
            content=new_code,
            root=root,
            allow_outside_root=allow_outside_root,
            state=state
        )
        return f"Successfully replaced file content in {path}"
    except Exception as e:
        return f"Failed to replace file content: {str(e)}"


def _extract_function_name(new_code: str) -> Optional[str]:
    try:
        tree = ast.parse(new_code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_GENERATION, TaskType.CODE_FIX)
def inject_function(
    path: str,
    new_code: str,
    function_name: str | None = None,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> str:
    """Insert a new top-level function into a file.

    Args:
        path: Path to the python file
        new_code: Complete new source code for the function
        function_name: Optional function name to validate against file contents
        root: Project root directory (defaults to cwd)
        allow_outside_root: Whether to allow writing outside root
        state: Agent state (optional)

    Returns:
        Success message or error description
    """
    if root is None:
        root = Path.cwd()

    is_valid, error_msg = _validate_with_dedent(new_code)
    if not is_valid:
        return f"Invalid Python syntax in injected code:\n{error_msg}\n\nCode attempted:\n{new_code}"

    inferred_name = _extract_function_name(new_code)
    target_name = function_name or inferred_name
    if not target_name:
        return "Injected code must define a top-level function."
    if function_name and inferred_name and function_name != inferred_name:
        return f"Injected function name '{inferred_name}' does not match '{function_name}'."

    try:
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        if find_function_lines(content, target_name):
            return f"Function '{target_name}' already exists in {path}"

        lines = content.splitlines(keepends=True)
        guard_index = None
        guard_pattern = re.compile(r"^\s*if __name__ == [\"']__main__[\"']\s*:")
        for idx, line in enumerate(lines):
            if guard_pattern.match(line):
                guard_index = idx
                break

        insert_line = (guard_index + 1) if guard_index is not None else (len(lines) + 1)
        snippet = new_code.rstrip("\n") + "\n\n"
        inject_lines(
            path=path,
            line=insert_line,
            content=snippet,
            root=root,
            allow_outside_root=allow_outside_root,
            state=state,
        )
        return f"Successfully injected function '{target_name}' into {path}"
    except Exception as e:
        return f"Failed to inject function: {str(e)}"
