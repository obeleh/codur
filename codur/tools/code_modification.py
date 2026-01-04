"""
High-level code modification tools for the coding agent.
These tools combine AST analysis with file modification to robustly update code entities.
"""

from pathlib import Path
from typing import Optional, Any, TypedDict
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
    summary_format,
    tool_contexts,
    tool_guards,
    tool_scenarios,
    tool_side_effects,
)
from codur.tools.validation import validate_python_syntax


def _validate_with_dedent(code: str) -> tuple[bool, Optional[str]]:
    """Validate syntax, attempting dedent if initial parse fails due to indentation."""
    result = validate_python_syntax(code)
    if not result.get("valid") and "unexpected indent" in str(result.get("error", "")):
        # Try dedenting
        dedented_code = textwrap.dedent(code)
        result_dedent = validate_python_syntax(dedented_code)
        if result_dedent.get("valid"):
            return True, None
    return result.get("valid", False), result.get("error")


class CodeModificationResult(TypedDict, total=False):
    """Result payload for code modification operations."""
    ok: bool
    operation: str
    path: str
    message: str
    target: str | None
    error: str | None
    start_line: int | None
    end_line: int | None
    inserted_line: int | None
    inserted_lines: int | None


def _build_result(
    *,
    ok: bool,
    operation: str,
    path: str,
    target: str | None,
    message: str,
    error: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    inserted_line: int | None = None,
    inserted_lines: int | None = None,
) -> CodeModificationResult:
    """Build a consistent result dict for code modification tools."""
    dct: CodeModificationResult = {
        "ok": ok,
        "operation": operation,
        "path": path,
        "message": message,
    }
    if target is not None:
        dct["target"] = target
    if error is not None:
        dct["error"] = error
    if start_line is not None:
        dct["start_line"] = start_line
    if end_line is not None:
        dct["end_line"] = end_line
    if inserted_line is not None:
        dct["inserted_line"] = inserted_line
    if inserted_lines is not None:
        dct["inserted_lines"] = inserted_lines
    return dct

@summary_format("replaced function <function_name> in <path>")
@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR)
def replace_function(
    path: str,
    function_name: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> CodeModificationResult:
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
        message = f"Invalid Python syntax in replacement code:\n{error_msg}\n\nCode attempted:\n{new_code}"
        return _build_result(
            ok=False,
            operation="replace_function",
            path=str(path),
            target=function_name,
            message=message,
            error=message,
        )

    try:
        # Read current content to find lines
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        
        line_range = find_function_lines(content, function_name)
        if not line_range:
            message = f"Could not find function '{function_name}' in {path}"
            return _build_result(
                ok=False,
                operation="replace_function",
                path=str(path),
                target=function_name,
                message=message,
                error=message,
            )
        
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
        
        message = f"Successfully replaced function '{function_name}' in {path}"
        return _build_result(
            ok=True,
            operation="replace_function",
            path=str(path),
            target=function_name,
            message=message,
            start_line=start_line,
            end_line=end_line,
        )

    except Exception as e:
        message = f"Failed to replace function: {str(e)}"
        return _build_result(
            ok=False,
            operation="replace_function",
            path=str(path),
            target=function_name,
            message=message,
            error=message,
        )


@summary_format("replaced class <class_name> in <path>")
@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.REFACTOR)
def replace_class(
    path: str,
    class_name: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> CodeModificationResult:
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
        message = f"Invalid Python syntax in replacement code:\n{error_msg}\n\nCode attempted:\n{new_code}"
        return _build_result(
            ok=False,
            operation="replace_class",
            path=str(path),
            target=class_name,
            message=message,
            error=message,
        )

    try:
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        
        line_range = find_class_lines(content, class_name)
        if not line_range:
            message = f"Could not find class '{class_name}' in {path}"
            return _build_result(
                ok=False,
                operation="replace_class",
                path=str(path),
                target=class_name,
                message=message,
                error=message,
            )
        
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
        
        message = f"Successfully replaced class '{class_name}' in {path}"
        return _build_result(
            ok=True,
            operation="replace_class",
            path=str(path),
            target=class_name,
            message=message,
            start_line=start_line,
            end_line=end_line,
        )

    except Exception as e:
        message = f"Failed to replace class: {str(e)}"
        return _build_result(
            ok=False,
            operation="replace_class",
            path=str(path),
            target=class_name,
            message=message,
            error=message,
        )


@summary_format("replaced method <class_name>.<method_name> in <path>")
@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.REFACTOR)
def replace_method(
    path: str,
    class_name: str,
    method_name: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> CodeModificationResult:
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
        message = f"Invalid Python syntax in replacement code:\n{error_msg}\n\nCode attempted:\n{new_code}"
        return _build_result(
            ok=False,
            operation="replace_method",
            path=str(path),
            target=f"{class_name}.{method_name}",
            message=message,
            error=message,
        )

    try:
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        
        line_range = find_method_lines(content, class_name, method_name)
        if not line_range:
            message = f"Could not find method '{class_name}.{method_name}' in {path}"
            return _build_result(
                ok=False,
                operation="replace_method",
                path=str(path),
                target=f"{class_name}.{method_name}",
                message=message,
                error=message,
            )
        
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
        
        message = f"Successfully replaced method '{class_name}.{method_name}' in {path}"
        return _build_result(
            ok=True,
            operation="replace_method",
            path=str(path),
            target=f"{class_name}.{method_name}",
            message=message,
            start_line=start_line,
            end_line=end_line,
        )

    except Exception as e:
        message = f"Failed to replace method: {str(e)}"
        return _build_result(
            ok=False,
            operation="replace_method",
            path=str(path),
            target=f"{class_name}.{method_name}",
            message=message,
            error=message,
        )


@summary_format("replaced content of file <path>")
@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_guards(ToolGuard.TEST_OVERWRITE)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION, TaskType.REFACTOR)
def replace_file_content(
    path: str,
    new_code: str,
    root: Path | None = None,
    allow_outside_root: bool = False,
    state: Any | None = None,
) -> CodeModificationResult:
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
        result = validate_python_syntax(new_code)
        if not result.get("valid"):
            error_msg = result.get("error", "Unknown error")
            message = f"Invalid Python syntax:\n{error_msg}\n\nCode attempted:\n{new_code}"
            return _build_result(
                ok=False,
                operation="replace_file_content",
                path=str(path),
                target=None,
                message=message,
                error=message,
            )

    try:
        write_file(
            path=path,
            content=new_code,
            root=root,
            allow_outside_root=allow_outside_root,
            state=state
        )
        message = f"Successfully replaced file content in {path}"
        return _build_result(
            ok=True,
            operation="replace_file_content",
            path=str(path),
            target=None,
            message=message,
        )
    except Exception as e:
        message = f"Failed to replace file content: {str(e)}"
        return _build_result(
            ok=False,
            operation="replace_file_content",
            path=str(path),
            target=None,
            message=message,
            error=message,
        )


def _extract_function_name(new_code: str) -> Optional[str]:
    """Return the first top-level function name from parsed code."""
    try:
        tree = ast.parse(new_code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


@summary_format("injected function <function_name> into <path>")
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
) -> CodeModificationResult:
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
        message = f"Invalid Python syntax in injected code:\n{error_msg}\n\nCode attempted:\n{new_code}"
        return _build_result(
            ok=False,
            operation="inject_function",
            path=str(path),
            target=function_name or _extract_function_name(new_code),
            message=message,
            error=message,
        )

    inferred_name = _extract_function_name(new_code)
    target_name = function_name or inferred_name
    if not target_name:
        message = "Injected code must define a top-level function."
        return _build_result(
            ok=False,
            operation="inject_function",
            path=str(path),
            target=None,
            message=message,
            error=message,
        )
    if function_name and inferred_name and function_name != inferred_name:
        message = f"Injected function name '{inferred_name}' does not match '{function_name}'."
        return _build_result(
            ok=False,
            operation="inject_function",
            path=str(path),
            target=function_name,
            message=message,
            error=message,
        )

    try:
        content = read_file(path=path, root=root, allow_outside_root=allow_outside_root, state=state)
        if find_function_lines(content, target_name):
            message = f"Function '{target_name}' already exists in {path}"
            return _build_result(
                ok=False,
                operation="inject_function",
                path=str(path),
                target=target_name,
                message=message,
                error=message,
            )

        lines = content.splitlines(keepends=True)
        guard_index = None
        guard_pattern = re.compile(r"^\s*if __name__ == [\"']__main__[\"']\s*:")
        for idx, line in enumerate(lines):
            if guard_pattern.match(line):
                guard_index = idx
                break

        insert_line = (guard_index + 1) if guard_index is not None else (len(lines) + 1)
        snippet = new_code.rstrip("\n") + "\n\n"
        insert_result = inject_lines(
            path=path,
            line=insert_line,
            content=snippet,
            root=root,
            allow_outside_root=allow_outside_root,
            state=state,
        )
        message = f"Successfully injected function '{target_name}' into {path}"
        return _build_result(
            ok=True,
            operation="inject_function",
            path=str(path),
            target=target_name,
            message=message,
            inserted_line=insert_result.get("line"),
            inserted_lines=insert_result.get("inserted_lines"),
        )
    except Exception as e:
        message = f"Failed to inject function: {str(e)}"
        return _build_result(
            ok=False,
            operation="inject_function",
            path=str(path),
            target=target_name,
            message=message,
            error=message,
        )
