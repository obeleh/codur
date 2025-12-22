"""AST-based utilities for finding line ranges of functions/classes."""

import ast
from typing import Optional


def find_function_lines(
    file_content: str,
    function_name: str
) -> Optional[tuple[int, int]]:
    """
    Find line range of a function definition in Python code.

    Args:
        file_content: Python source code
        function_name: Name of function to find

    Returns:
        (start_line, end_line) as 1-based line numbers, or None if not found
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # AST uses 1-based line numbers, which matches our tools
            start_line = node.lineno
            # end_lineno available in Python 3.8+
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            return (start_line, end_line)

    return None


def find_class_lines(
    file_content: str,
    class_name: str
) -> Optional[tuple[int, int]]:
    """
    Find line range of a class definition in Python code.

    Args:
        file_content: Python source code
        class_name: Name of class to find

    Returns:
        (start_line, end_line) as 1-based line numbers, or None if not found
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            return (start_line, end_line)

    return None


def find_method_lines(
    file_content: str,
    class_name: str,
    method_name: str
) -> Optional[tuple[int, int]]:
    """
    Find line range of a method within a class.

    Args:
        file_content: Python source code
        class_name: Name of containing class
        method_name: Name of method to find

    Returns:
        (start_line, end_line) as 1-based line numbers, or None if not found
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    start_line = item.lineno
                    end_line = item.end_lineno if hasattr(item, 'end_lineno') else item.lineno
                    return (start_line, end_line)

    return None
