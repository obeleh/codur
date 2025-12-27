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
    config: Optional[CodurConfig] = None,
    state: Optional[AgentState] = None,
) -> str:
    """Execute a Python file and return its output.

    This tool allows the LLM to run and validate code during the coding phase.
    It executes the specified Python file and captures stdout/stderr.

    Args:
        path: Path to the Python file to execute (relative to root)
        root: Root directory for relative paths (defaults to current directory)
        config: Codur configuration object (injected by tool executor)
        state: Current agent state (injected by tool executor)

    Returns:
        String with the output from running the file, or error message if execution failed.

    Example:
        run_python_file("main.py")
        # Returns the output from executing main.py
    """
    cwd = Path(root) if root != "." else Path.cwd()
    file_path = cwd / path

    if not file_path.exists():
        return f"Error: File not found: {file_path}"

    if not file_path.is_file():
        return f"Error: Not a file: {file_path}"

    execution_timeout = 60

    try:
        process = subprocess.Popen(
            ["python", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd)
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
