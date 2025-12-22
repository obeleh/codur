"""Python syntax validation utilities."""

import ast
from typing import Optional


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
