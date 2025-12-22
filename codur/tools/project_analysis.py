"""
Project analysis tools for Codur.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from codur.constants import DEFAULT_MAX_RESULTS
from codur.graph.state import AgentState
from codur.tools.filesystem import EXCLUDE_DIRS
from codur.utils.path_utils import resolve_path, resolve_root

DEFAULT_MAX_NODES = 2000
DEFAULT_MAX_EDGES = 4000


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(dirpath) / filename)
    return files


def _module_name_for_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.stem

    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        module_parts = parts[:-1]
        return ".".join(module_parts) if module_parts else "__init__"
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _resolve_internal_module(name: str, internal_modules: set[str]) -> str | None:
    if not name:
        return None
    if name in internal_modules:
        return name
    parts = name.split(".")
    for idx in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:idx])
        if candidate in internal_modules:
            return candidate
    return None


def _base_package(current_module: str, is_package: bool, level: int) -> str:
    if is_package:
        package = current_module
    else:
        package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    if level <= 1:
        return package
    parts = package.split(".") if package else []
    keep = max(0, len(parts) - (level - 1))
    return ".".join(parts[:keep])


def python_dependency_graph(
    root: str | Path | None = None,
    paths: list[str] | None = None,
    include_external: bool = False,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Build a Python module dependency graph and return DOT output.
    """
    root_path = resolve_root(root)
    file_paths: list[Path] = []

    if paths:
        for raw in paths:
            target = resolve_path(raw, root_path, allow_outside_root=allow_outside_root)
            if target.is_dir():
                file_paths.extend(_iter_python_files(target))
            elif target.is_file() and target.suffix == ".py":
                file_paths.append(target)
    else:
        file_paths = _iter_python_files(root_path)

    module_map: dict[str, Path] = {}
    for file_path in sorted(set(file_paths)):
        module_name = _module_name_for_path(file_path, root_path)
        module_map[module_name] = file_path

    internal_modules = set(module_map.keys())
    external_modules: set[str] = set()
    edges: set[tuple[str, str]] = set()
    parse_errors: list[dict] = []

    for module_name, file_path in module_map.items():
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            parse_errors.append({
                "file": str(file_path),
                "line": exc.lineno or 0,
                "column": exc.offset or 0,
                "message": exc.msg,
            })
            continue
        except OSError as exc:
            parse_errors.append({
                "file": str(file_path),
                "line": 0,
                "column": 0,
                "message": f"Failed to read file: {exc}",
            })
            continue

        is_package = file_path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = _resolve_internal_module(alias.name, internal_modules)
                    if resolved:
                        if resolved != module_name:
                            edges.add((module_name, resolved))
                    elif include_external:
                        external_modules.add(alias.name)
                        edges.add((module_name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                base = _base_package(module_name, is_package, level)
                module_part = node.module or ""
                base_module = f"{base}.{module_part}" if base and module_part else (module_part or base)
                for alias in node.names:
                    if alias.name == "*":
                        candidates = [base_module]
                    else:
                        full = f"{base_module}.{alias.name}" if base_module else alias.name
                        candidates = [full, base_module, alias.name]
                    resolved = None
                    for candidate in candidates:
                        resolved = _resolve_internal_module(candidate, internal_modules)
                        if resolved:
                            break
                    if resolved:
                        if resolved != module_name:
                            edges.add((module_name, resolved))
                    elif include_external:
                        external_name = base_module or alias.name
                        if external_name:
                            external_modules.add(external_name)
                            edges.add((module_name, external_name))

    nodes = sorted(internal_modules | external_modules) if include_external else sorted(internal_modules)
    truncated_nodes = False
    if max_nodes and len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
        truncated_nodes = True

    node_set = set(nodes)
    edges_list = sorted(edge for edge in edges if edge[0] in node_set and edge[1] in node_set)
    truncated_edges = False
    if max_edges and len(edges_list) > max_edges:
        edges_list = edges_list[:max_edges]
        truncated_edges = True

    external_included = sorted(name for name in external_modules if name in node_set) if include_external else []

    dot_lines = [
        "digraph dependencies {",
        "  rankdir=LR;",
        "  node [shape=box];",
    ]
    for name in external_included:
        dot_lines.append(f'  "{name}" [shape=ellipse, style=dashed];')
    for source, target in edges_list:
        dot_lines.append(f'  "{source}" -> "{target}";')
    dot_lines.append("}")

    return {
        "root": str(root_path),
        "files": len(file_paths),
        "node_count": len(node_set),
        "edge_count": len(edges_list),
        "truncated": {"nodes": truncated_nodes, "edges": truncated_edges},
        "nodes": nodes,
        "external_nodes": external_included,
        "edges": [{"from": src, "to": dst} for src, dst in edges_list],
        "dot": "\n".join(dot_lines),
        "errors": parse_errors[:DEFAULT_MAX_RESULTS],
    }
