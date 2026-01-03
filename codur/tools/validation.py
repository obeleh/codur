"""Python syntax validation utilities and code execution verification."""

import ast
import os
import subprocess
from pathlib import Path
from typing import Optional, TypedDict

from codur.config import CodurConfig
from codur.graph.state_operations import is_verbose, get_config


class ValidateSyntaxResult(TypedDict, total=False):
    """Result from validate_python_syntax."""
    valid: bool
    error: str


class RunPythonFileResult(TypedDict, total=False):
    """Result from run_python_file."""
    std_out: str
    error: str
    return_code: int
    std_err: str | None

class RunPytestResult(TypedDict, total=False):
    """Result from run_pytest."""
    success: bool
    exit_code: int | None
    command: str
    cwd: str
    paths: list[str]
    stdout: str
    stderr: str
    error: str


from codur.constants import DEFAULT_MAX_BYTES, TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import (
    ToolContext,
    ToolSideEffect,
    tool_contexts,
    tool_scenarios,
    tool_side_effects,
)
from codur.utils.config_helpers import get_cli_timeout
from codur.utils.path_utils import resolve_path, resolve_root
from codur.utils.text_helpers import truncate_chars
from codur.utils.validation import require_directory_exists


@tool_scenarios(
    TaskType.CODE_FIX,
    TaskType.CODE_GENERATION,
    TaskType.CODE_VALIDATION,
    TaskType.REFACTOR,
)
def validate_python_syntax(code: str) -> ValidateSyntaxResult:
    """
    Validate Python syntax using AST parsing.

    Args:
        code: Python source code to validate

    Returns:
        {"valid": True} if valid
        {"valid": False, "error": error_message} if invalid
    """
    try:
        ast.parse(code)
        return {"valid": True}
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n{e.text}"
            if e.offset:
                error_msg += "\n" + " " * (e.offset - 1) + "^"
        return {"valid": False, "error": error_msg}
    except Exception as e:
        return {"valid": False, "error": f"Parse error: {str(e)}"}


@tool_side_effects(ToolSideEffect.CODE_EXECUTION)
@tool_scenarios(TaskType.CODE_VALIDATION, TaskType.CODE_FIX)
def run_python_file(
    path: str,
    root: str = ".",
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    config: Optional[CodurConfig] = None,
    state: Optional[AgentState] = None,
) -> RunPythonFileResult:
    """Execute a Python file and return structured output.

    This tool allows the LLM to run and validate code during the coding phase.
    It executes the specified Python file and captures stdout/stderr.

    Args:
        path: Path to the Python file to execute (relative to root)
        root: Root directory for relative paths (defaults to current directory)
        cwd: Working directory for execution (defaults to root)
        env: Environment variables as a dict (merged with current environment)
        config: Codur configuration object (injected by tool executor)
        state: Current agent state (injected by tool executor)

    Returns:
        RunPythonFileResult with std_out/std_err/return_code.
        The "error" field is reserved for tool execution failures.

    Examples:
        run_python_file("main.py")
        # Returns std_out/std_err/return_code for main.py

        run_python_file("main.py", cwd="/tmp", env={"TIMEOUT": "30"})
        # Executes in /tmp with custom environment variables
    """
    root_dir = Path(root) if root != "." else Path.cwd()
    file_path = root_dir / path

    # Determine execution directory
    exec_cwd = Path(cwd) if cwd else root_dir

    if not file_path.exists():
        return {
            "error": f"Error: File not found: {file_path}"
        }

    if not file_path.is_file():
        return {
            "error": f"Error: Not a file: {file_path}"
        }

    execution_timeout = 60

    try:
        # Build environment: start with current environment and merge in custom vars
        process_env = dict(os.environ)
        if env:
            process_env.update(env)

        process = subprocess.Popen(
            ["python", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(exec_cwd),
            env=process_env
        )

        try:
            stdout, stderr = process.communicate(timeout=execution_timeout)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {
                "error": f"Error: Execution timed out after {execution_timeout} seconds"
            }

        std_out = stdout.strip() if stdout else ""
        std_err = stderr.strip() if stderr else ""

        if is_verbose(state):
            print(f"[dim] Return code: {return_code} Stdout:\n{std_out} [/dim]")
            if std_err:
                print(f"[yellow] Stderr: {std_err}[/yellow]")

        result: RunPythonFileResult = {
            "return_code": return_code,
            "std_out": std_out,
            "std_err": std_err if std_err else None,
        }
        return result

    except Exception as e:
        return {
            "error": f"Error: {str(e)}"
        }


@tool_side_effects(ToolSideEffect.CODE_EXECUTION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_VALIDATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def run_pytest(
    path: str | None = None,
    paths: list[str] | None = None,
    keyword: str | None = None,
    markers: str | None = None,
    extra_args: list[str] | None = None,
    root: str | Path | None = None,
    cwd: str | None = None,
    env: dict | None = None,
    timeout: int | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> RunPytestResult:
    """Run pytest and return the results."""
    if path and paths:
        raise ValueError("Specify either 'path' or 'paths', not both.")

    if path:
        paths = [path]

    root_path = resolve_root(root)
    exec_cwd = (
        resolve_path(cwd, root_path, allow_outside_root=allow_outside_root)
        if cwd
        else root_path
    )
    require_directory_exists(exec_cwd, context="pytest cwd")

    cmd: list[str] = ["pytest"]
    if keyword:
        cmd += ["-k", keyword]
    if markers:
        cmd += ["-m", markers]
    if extra_args:
        cmd.extend(extra_args)

    resolved_paths: list[str] = []
    if paths:
        for raw_path in paths:
            target = resolve_path(raw_path, root_path, allow_outside_root=allow_outside_root)
            if not target.exists():
                raise ValueError(f"Path does not exist: {raw_path}")
            resolved_paths.append(str(target))
        cmd.extend(resolved_paths)

    process_env = dict(os.environ)
    if env:
        process_env.update(env)

    config = get_config(state)
    effective_timeout = int(timeout) if timeout is not None else get_cli_timeout(config)
    max_output_chars = (
        int(getattr(getattr(config, "tools", None), "default_max_bytes", DEFAULT_MAX_BYTES))
        if config is not None
        else DEFAULT_MAX_BYTES
    )

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(exec_cwd),
            env=process_env,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "exit_code": 127,
            "error": "pytest not found on PATH",
            "command": " ".join(cmd),
            "cwd": str(exec_cwd),
        }

    try:
        stdout, stderr = process.communicate(timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return {
            "success": False,
            "exit_code": None,
            "error": f"Execution timed out after {effective_timeout} seconds",
            "command": " ".join(cmd),
            "cwd": str(exec_cwd),
        }

    stdout = truncate_chars(stdout or "", max_output_chars).strip()
    stderr = truncate_chars(stderr or "", max_output_chars).strip()
    return {
        "success": process.returncode == 0,
        "exit_code": process.returncode,
        "command": " ".join(cmd),
        "cwd": str(exec_cwd),
        "paths": resolved_paths,
        "stdout": stdout,
        "stderr": stderr,
    }
