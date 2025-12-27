"""Python syntax validation utilities and code execution verification."""

import ast
import subprocess
import os
from pathlib import Path
from typing import Optional

from codur.config import CodurConfig
from codur.graph.state import AgentState


def validate_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python syntax using AST parsing.

    Args:
        code: Python source code to validate

    Returns:
        (True, None) if valid
        (False, error_message) if invalid
    """
    try:
        ast.parse(code)
        return (True, None)
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n{e.text}"
            if e.offset:
                error_msg += "\n" + " " * (e.offset - 1) + "^"
        return (False, error_msg)
    except Exception as e:
        return (False, f"Parse error: {str(e)}")


def run_python_file(
    path: str,
    root: str = ".",
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    config: Optional[CodurConfig] = None,
    state: Optional[AgentState] = None,
) -> str:
    """Execute a Python file and return its output.

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
        String with the output from running the file, or error message if execution failed.

    Examples:
        run_python_file("main.py")
        # Returns the output from executing main.py

        run_python_file("main.py", cwd="/tmp", env={"TIMEOUT": "30"})
        # Executes in /tmp with custom environment variables
    """
    root_dir = Path(root) if root != "." else Path.cwd()
    file_path = root_dir / path

    # Determine execution directory
    exec_cwd = Path(cwd) if cwd else root_dir

    if not file_path.exists():
        return f"Error: File not found: {file_path}"

    if not file_path.is_file():
        return f"Error: Not a file: {file_path}"

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
            return f"Error: Execution timed out after {execution_timeout} seconds"

        # Format result
        output = stdout.strip() if stdout else ""

        if return_code != 0:
            error_msg = stderr.strip() if stderr else "Unknown error"
            if output:
                return f"Error (exit code {return_code}):\n{error_msg}\n\nOutput:\n{output}"
            else:
                return f"Error (exit code {return_code}):\n{error_msg}"

        return output if output else "(No output)"

    except Exception as e:
        return f"Error: {str(e)}"
