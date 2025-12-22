"""
Python AST tools for Codur.
"""

from __future__ import annotations

import ast
from pathlib import Path

from codur.constants import DEFAULT_MAX_RESULTS
from codur.graph.state import AgentState
from codur.utils.path_utils import resolve_path

DEFAULT_AST_MAX_NODES = 2000


def _node_label(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.Constant):
        text = repr(node.value)
        if len(text) > 80:
            return text[:77] + "..."
        return text
    return ""


def _node_location(node: ast.AST) -> dict:
    info = {}
    if hasattr(node, "lineno"):
        info["lineno"] = getattr(node, "lineno")
    if hasattr(node, "col_offset"):
        info["col_offset"] = getattr(node, "col_offset")
    if hasattr(node, "end_lineno"):
        info["end_lineno"] = getattr(node, "end_lineno")
    if hasattr(node, "end_col_offset"):
        info["end_col_offset"] = getattr(node, "end_col_offset")
    return info


def python_ast_graph(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    max_nodes: int = DEFAULT_AST_MAX_NODES,
    state: AgentState | None = None,
) -> dict:
    """
    Return a node/edge AST graph for a Python file.
    """
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        source = handle.read()

    tree = ast.parse(source, filename=str(target))
    nodes: list[dict] = []
    edges: list[dict] = []
    stack: list[tuple[ast.AST, int | None]] = [(tree, None)]
    node_id = 0
    truncated = False

    while stack:
        node, parent_id = stack.pop()
        current_id = node_id
        node_id += 1

        info = {
            "id": current_id,
            "type": node.__class__.__name__,
        }
        label = _node_label(node)
        if label:
            info["label"] = label
        info.update(_node_location(node))
        nodes.append(info)

        if parent_id is not None:
            edges.append({"from": parent_id, "to": current_id})

        if max_nodes and len(nodes) >= max_nodes:
            truncated = True
            break

        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            stack.append((child, current_id))

    return {
        "file": str(target),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "truncated": truncated,
        "nodes": nodes,
        "edges": edges,
    }


def _format_args(args: ast.arguments) -> list[str]:
    items: list[str] = []
    for arg in args.posonlyargs:
        items.append(arg.arg)
    if args.posonlyargs:
        items.append("/")
    for arg in args.args:
        items.append(arg.arg)
    if args.vararg:
        items.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        items.append("*")
    for arg in args.kwonlyargs:
        items.append(arg.arg)
    if args.kwarg:
        items.append(f"**{args.kwarg.arg}")
    return items


def _safe_unparse(node: ast.AST) -> str:
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            return node.__class__.__name__
    return ast.dump(node, include_attributes=False)


def python_ast_outline(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    include_docstrings: bool = False,
    include_decorators: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
    state: AgentState | None = None,
) -> dict:
    """
    Return a high-level outline of classes and functions in a Python file.
    """
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        source = handle.read()

    tree = ast.parse(source, filename=str(target))
    results: list[dict] = []
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    truncated = False

    while stack:
        node, parent_qual = stack.pop()
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        next_parent = parent_qual
        if is_def:
            qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
            entry = {
                "type": (
                    "class"
                    if isinstance(node, ast.ClassDef)
                    else "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                ),
                "name": node.name,
                "qualname": qual,
                "lineno": getattr(node, "lineno", None),
                "end_lineno": getattr(node, "end_lineno", None),
            }
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                entry["args"] = _format_args(node.args)
            if include_docstrings:
                entry["docstring"] = ast.get_docstring(node) or ""
            if include_decorators and getattr(node, "decorator_list", None):
                entry["decorators"] = [_safe_unparse(dec) for dec in node.decorator_list]
            results.append(entry)
            next_parent = qual

            if max_results and len(results) >= max_results:
                truncated = True
                break

        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            stack.append((child, next_parent))

    return {
        "file": str(target),
        "count": len(results),
        "truncated": truncated,
        "definitions": results,
    }
