"""
Project analysis tools for Codur.
"""

from __future__ import annotations

import ast
import os
import sys
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from typing import Iterator

from codur.constants import DEFAULT_MAX_RESULTS, TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import ToolContext, tool_contexts, tool_scenarios
from codur.utils.ignore_utils import (
    get_config_from_state,
    get_exclude_dirs,
    is_gitignored,
    load_gitignore,
    should_include_hidden,
    should_respect_gitignore,
)
from codur.utils.path_utils import resolve_path, resolve_root
from codur.utils.validation import validate_file_access

DEFAULT_MAX_NODES = 2000
DEFAULT_MAX_EDGES = 4000


def _iter_python_files(
    root: Path,
    exclude_folders: list[str] | None = None,
    config: object | None = None,
) -> list[Path]:
    files: list[Path] = []
    default_excludes = get_exclude_dirs(config)
    include_hidden = should_include_hidden(config)
    gitignore_spec = load_gitignore(root) if should_respect_gitignore(config) else None
    
    # Pre-process exclude folders for platform compatibility
    norm_excludes = []
    if exclude_folders:
        norm_excludes = [ex.replace("/", os.sep) for ex in exclude_folders]

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        filtered_dirs: list[str] = []
        for dirname in dirnames:
            if dirname in default_excludes:
                continue
            if not include_hidden and dirname.startswith("."):
                continue
            rel_path = rel_dir / dirname
            if gitignore_spec and is_gitignored(rel_path, root, gitignore_spec, is_dir=True):
                continue
            if norm_excludes:
                s_rel = str(rel_path)
                if any(s_rel == ex or s_rel.startswith(ex + os.sep) for ex in norm_excludes):
                    continue
            filtered_dirs.append(dirname)
        dirnames[:] = filtered_dirs

        for filename in filenames:
            if not include_hidden and filename.startswith("."):
                continue
            rel_path = rel_dir / filename
            if gitignore_spec and is_gitignored(rel_path, root, gitignore_spec, is_dir=False):
                continue
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
    if level == 0:
        return ""
    if is_package:
        package = current_module
    else:
        package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    if level <= 1:
        return package
    parts = package.split(".") if package else []
    keep = max(0, len(parts) - (level - 1))
    return ".".join(parts[:keep])


def _is_excluded_module(name: str, excludes: list[str] | None) -> bool:
    if not excludes:
        return False
    for ex in excludes:
        if name == ex or name.startswith(ex + "."):
            return True
    return False


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.EXPLANATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def python_dependency_graph(
    root: str | Path | None = None,
    paths: list[str] | None = None,
    include_external: bool = False,
    include_styling: bool = False,
    exclude_modules: list[str] | None = None,
    exclude_folders: list[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Build a Python module dependency graph and return DOT output.

    Args:
        root: Root directory to analyze (defaults to current working directory)
        paths: Specific paths to analyze (files or directories). If None, analyzes entire root
        include_external: Include external (non-project) module dependencies
        include_styling: Add visual styling to DOT output (shapes, colors, etc)
        exclude_modules: Module names or prefixes to exclude from the graph
        exclude_folders: Folder names to exclude from scanning
        max_nodes: Maximum number of nodes to include (0 for unlimited)
        max_edges: Maximum number of edges to include (0 for unlimited)
        allow_outside_root: Allow analyzing paths outside the root directory
        state: Agent state for configuration (internal)
    """
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    file_paths: list[Path] = []

    if paths:
        for raw in paths:
            target = resolve_path(raw, root_path, allow_outside_root=allow_outside_root)
            if target.is_dir():
                file_paths.extend(_iter_python_files(target, exclude_folders=exclude_folders, config=config))
            elif target.is_file() and target.suffix == ".py":
                file_paths.append(target)
    else:
        file_paths = _iter_python_files(root_path, exclude_folders=exclude_folders, config=config)

    module_map: dict[str, Path] = {}
    for file_path in sorted(set(file_paths)):
        module_name = _module_name_for_path(file_path, root_path)
        if exclude_modules and _is_excluded_module(module_name, exclude_modules):
            continue
        module_map[module_name] = file_path

    internal_modules = set(module_map.keys())
    external_modules: set[str] = set()
    edges: set[tuple[str, str]] = set()
    parse_errors: list[dict] = []

    for module_name, file_path in module_map.items():
        try:
            validate_file_access(
                file_path,
                root_path,
                config,
                operation="read",
                allow_outside_root=allow_outside_root,
            )
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
                        if resolved != module_name and not (exclude_modules and _is_excluded_module(resolved, exclude_modules)):
                            edges.add((module_name, resolved))
                    elif include_external:
                        if not (exclude_modules and _is_excluded_module(alias.name, exclude_modules)):
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
                        if resolved != module_name and not (exclude_modules and _is_excluded_module(resolved, exclude_modules)):
                            edges.add((module_name, resolved))
                    elif include_external:
                        external_name = base_module or alias.name
                        if external_name and not (exclude_modules and _is_excluded_module(external_name, exclude_modules)):
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

    dot_lines = ["digraph dependencies {"]
    if include_styling:
        dot_lines.extend([
            "  rankdir=LR;",
            "  node [shape=box];",
        ])
    for name in external_included:
        style = ' [shape=ellipse, style=dashed]' if include_styling else ""
        dot_lines.append(f'  "{name}"{style};')
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


class _DeepGraphVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module_name: str,
        internal_modules: set[str],
        include_external: bool,
        is_package: bool,
    ):
        self.module_name = module_name
        self.internal_modules = internal_modules
        self.include_external = include_external
        self.is_package = is_package
        self.nodes: list[dict] = []
        self.edges: set[tuple[str, str, str]] = set()  # (source, target, type)
        self.scope_stack: list[str] = [module_name]
        self.imports: dict[str, str] = {}  # alias -> full_resolved_name

        # Add module node
        self.nodes.append({
            "id": module_name,
            "type": "module",
            "name": module_name,
            "parent": None
        })

    def _resolve_name(self, name: str) -> str | None:
        if name in self.imports:
            return self.imports[name]
        # Basic check if it matches a top-level internal module
        if name in self.internal_modules:
            return name
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            resolved = _resolve_internal_module(alias.name, self.internal_modules)
            if resolved:
                self.imports[alias.asname or alias.name] = resolved
                self.edges.add((self.module_name, resolved, "import"))
            elif self.include_external:
                self.imports[alias.asname or alias.name] = alias.name
                self.edges.add((self.module_name, alias.name, "import_external"))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        level = node.level or 0
        base = _base_package(self.module_name, self.is_package, level)
        module_part = node.module or ""
        base_module = f"{base}.{module_part}" if base and module_part else (module_part or base)

        for alias in node.names:
            if alias.name == "*":
                # Wildcard imports are hard to resolve statically without full project index
                continue
            
            # Attempt to resolve the full name
            full = f"{base_module}.{alias.name}" if base_module else alias.name
            candidates = [full, base_module, alias.name]
            
            resolved = None
            for candidate in candidates:
                resolved = _resolve_internal_module(candidate, self.internal_modules)
                if resolved:
                    break
            
            target_name = alias.asname or alias.name
            if resolved:
                self.imports[target_name] = resolved
                self.edges.add((self.module_name, resolved, "import"))
            elif self.include_external:
                external_name = base_module or alias.name
                if external_name:
                    self.imports[target_name] = external_name
                    self.edges.add((self.module_name, external_name, "import_external"))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        node_id = f"{self.scope_stack[-1]}.{node.name}"
        self.nodes.append({
            "id": node_id,
            "type": "class",
            "name": node.name,
            "parent": self.scope_stack[-1]
        })
        
        # Inheritance edges
        for base in node.bases:
            if isinstance(base, ast.Name):
                target = self._resolve_name(base.id)
                if target:
                    self.edges.add((node_id, target, "inherits"))
            elif isinstance(base, ast.Attribute):
                # Try to resolve module.Class
                if isinstance(base.value, ast.Name):
                    module_alias = base.value.id
                    if module_alias in self.imports:
                        target = f"{self.imports[module_alias]}.{base.attr}"
                        # We might not know if this target is valid internally without a full symbol table
                        # but we record the edge anyway
                        self.edges.add((node_id, target, "inherits"))

        self.scope_stack.append(node_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_type = "module"
        # Heuristic: if parent ID contains more dots than module name, likely inside class/func
        if self.scope_stack[-1] != self.module_name:
            parent_type = "class" # Simplified assumption, could be nested func

        node_id = f"{self.scope_stack[-1]}.{node.name}"
        self.nodes.append({
            "id": node_id,
            "type": "method" if parent_type == "class" else "function",
            "name": node.name,
            "parent": self.scope_stack[-1]
        })
        
        self.scope_stack.append(node_id)
        self.generic_visit(node)
        self.scope_stack.pop()
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)


def deep_python_dependency_graph(
    root: str | Path | None = None,
    paths: list[str] | None = None,
    include_external: bool = False,
    include_styling: bool = False,
    exclude_modules: list[str] | None = None,
    exclude_folders: list[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Build a deep dependency graph including classes, methods, and functions.

    Args:
        root: Root directory to analyze (defaults to current working directory)
        paths: Specific paths to analyze (files or directories). If None, analyzes entire root
        include_external: Include external (non-project) module dependencies
        include_styling: Add visual styling to DOT output (shapes, colors, edge labels)
        exclude_modules: Module names or prefixes to exclude from the graph
        exclude_folders: Folder names to exclude from scanning
        max_nodes: Maximum number of nodes to include (0 for unlimited)
        max_edges: Maximum number of edges to include (0 for unlimited)
        allow_outside_root: Allow analyzing paths outside the root directory
        state: Agent state for configuration (internal)
    """
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    file_paths: list[Path] = []

    if paths:
        for raw in paths:
            target = resolve_path(raw, root_path, allow_outside_root=allow_outside_root)
            if target.is_dir():
                file_paths.extend(_iter_python_files(target, exclude_folders=exclude_folders, config=config))
            elif target.is_file() and target.suffix == ".py":
                file_paths.append(target)
    else:
        file_paths = _iter_python_files(root_path, exclude_folders=exclude_folders, config=config)

    module_map: dict[str, Path] = {}
    for file_path in sorted(set(file_paths)):
        module_name = _module_name_for_path(file_path, root_path)
        if exclude_modules and _is_excluded_module(module_name, exclude_modules):
            continue
        module_map[module_name] = file_path

    internal_modules = set(module_map.keys())
    
    all_nodes = []
    all_edges_set: set[tuple[str, str, str]] = set()
    parse_errors = []

    for module_name, file_path in module_map.items():
        try:
            validate_file_access(
                file_path,
                root_path,
                config,
                operation="read",
                allow_outside_root=allow_outside_root,
            )
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, OSError) as exc:
            msg = getattr(exc, "msg", str(exc))
            lineno = getattr(exc, "lineno", 0)
            parse_errors.append({
                "file": str(file_path),
                "message": msg,
                "line": lineno
            })
            continue

        is_package = file_path.name == "__init__.py"
        visitor = _DeepGraphVisitor(module_name, internal_modules, include_external, is_package)
        visitor.visit(tree)
        
        all_nodes.extend(visitor.nodes)
        all_edges_set.update(visitor.edges)

    # Process nodes and edges for output
    # Filter nodes based on exclude_modules
    if exclude_modules:
        all_nodes = [n for n in all_nodes if not _is_excluded_module(n["id"], exclude_modules)]

    # Filter if too many
    truncated_nodes = False
    if max_nodes and len(all_nodes) > max_nodes:
        all_nodes = all_nodes[:max_nodes]
        truncated_nodes = True
    
    node_ids = {n["id"] for n in all_nodes}
    
    # Convert set back to list of dicts and filter
    all_edges = []
    for s, t, ty in sorted(all_edges_set):
        # Exclude edges where target is an excluded module
        if exclude_modules and _is_excluded_module(t, exclude_modules):
            continue
        all_edges.append({"source": s, "target": t, "type": ty})
    
    filtered_edges = [e for e in all_edges if e["source"] in node_ids]
    truncated_edges = False
    if max_edges and len(filtered_edges) > max_edges:
        filtered_edges = filtered_edges[:max_edges]
        truncated_edges = True

    external_targets = sorted({e["target"] for e in filtered_edges if e["target"] not in node_ids})

    dot_lines = ["digraph dependencies {"]
    if include_styling:
        dot_lines.append("  rankdir=LR;")
    node_shapes = {
        "module": "box",
        "class": "ellipse",
        "function": "oval",
        "method": "oval",
    }
    for node in all_nodes:
        if include_styling:
            shape = node_shapes.get(node["type"], "box")
            dot_lines.append(f'  "{node["id"]}" [shape={shape}];')
        else:
            dot_lines.append(f'  "{node["id"]}";')
    for name in external_targets:
        style = " [shape=ellipse, style=dashed]" if include_styling else ""
        dot_lines.append(f'  "{name}"{style};')
    for edge in filtered_edges:
        attrs = []
        if include_styling and edge.get("type"):
            attrs.append(f'label="{edge["type"]}"')
        if include_styling and edge.get("type") == "import_external":
            attrs.append("style=dashed")
        attr_str = f" [{', '.join(attrs)}]" if attrs else ""
        dot_lines.append(f'  "{edge["source"]}" -> "{edge["target"]}"{attr_str};')
    dot_lines.append("}")

    return {
        "root": str(root_path),
        "files": len(file_paths),
        "node_count": len(node_ids),
        "edge_count": len(filtered_edges),
        "truncated": {"nodes": truncated_nodes, "edges": truncated_edges},
        "nodes": all_nodes,
        "external_nodes": external_targets,
        "edges": filtered_edges,
        "dot": "\n".join(dot_lines),
        "errors": parse_errors[:DEFAULT_MAX_RESULTS],
    }


def python_unused_code(
    root: str | Path | None = None,
    paths: list[str] | None = None,
    exclude_modules: list[str] | None = None,
    exclude_folders: list[str] | None = None,
    min_confidence: int = 60,
    sort_by_size: bool = False,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Identify unused code using vulture.

    Args:
        root: Root directory to analyze (defaults to current working directory)
        paths: Specific paths to analyze (files or directories). If None, analyzes entire root
        exclude_modules: Module names or prefixes to exclude from analysis
        exclude_folders: Folder names to exclude from scanning
        min_confidence: Minimum confidence level (0-100) for reporting unused code
        sort_by_size: Sort results by code size instead of alphabetically
        allow_outside_root: Allow analyzing paths outside the root directory
        state: Agent state for configuration (internal)
    """
    root_path = resolve_root(root)
    config = get_config_from_state(state)
    file_paths: list[Path] = []

    if paths:
        for raw in paths:
            target = resolve_path(raw, root_path, allow_outside_root=allow_outside_root)
            if target.is_dir():
                file_paths.extend(_iter_python_files(target, exclude_folders=exclude_folders, config=config))
            elif target.is_file() and target.suffix == ".py":
                file_paths.append(target)
    else:
        file_paths = _iter_python_files(root_path, exclude_folders=exclude_folders, config=config)

    module_map: dict[str, Path] = {}
    for file_path in sorted(set(file_paths)):
        module_name = _module_name_for_path(file_path, root_path)
        if exclude_modules and _is_excluded_module(module_name, exclude_modules):
            continue
        module_map[module_name] = file_path

    try:
        from vulture import Vulture
    except Exception as exc:
        return {
            "root": str(root_path),
            "files": len(file_paths),
            "min_confidence": min_confidence,
            "unused_items": [],
            "errors": [
                {"message": f"Failed to import vulture: {exc}"},
            ],
        }

    vulture = Vulture()
    vulture.scavenge([str(path) for path in module_map.values()])
    unused_items = []
    for item in vulture.get_unused_code(min_confidence=min_confidence, sort_by_size=sort_by_size):
        file_path = Path(item.filename)
        module_name = _module_name_for_path(file_path, root_path)
        if exclude_modules and _is_excluded_module(module_name, exclude_modules):
            continue
        unused_items.append({
            "name": item.name,
            "type": item.typ,
            "file": str(file_path),
            "line": item.first_lineno,
            "message": item.message,
            "confidence": item.confidence,
            "module": module_name,
            "size": item.size,
        })

    return {
        "root": str(root_path),
        "files": len(file_paths),
        "min_confidence": min_confidence,
        "unused_items": unused_items,
        "errors": [],
    }


@contextmanager
def _temporary_argv(args: list[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = original


def _serialize_summary(summary: dict) -> dict:
    serialized = {}
    for key, value in summary.items():
        if hasattr(value, "isoformat"):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


def _serialize_message(message, root_path: Path, absolute_paths: bool) -> dict:
    loc = message.location
    path_obj = None
    absolute_path = None
    if loc and loc.path:
        absolute_path = str(loc.path)
        if absolute_paths:
            path_obj = loc.path
        else:
            try:
                path_obj = loc.relative_path(root_path)
            except ValueError:
                path_obj = loc.path
    return {
        "source": message.source,
        "code": message.code,
        "message": message.message,
        "path": str(path_obj) if path_obj else None,
        "absolute_path": absolute_path,
        "module": loc.module if loc else None,
        "function": loc.function if loc else None,
        "line": loc.line if loc else None,
        "column": loc.character if loc else None,
        "line_end": loc.line_end if loc else None,
        "column_end": loc.character_end if loc else None,
        "doc_url": getattr(message, "doc_url", None),
        "fixable": getattr(message, "is_fixable", False),
    }


def _build_prospector_args(
    target_paths: list[str],
    *,
    tools: list[str] | None,
    with_tools: list[str] | None,
    without_tools: list[str] | None,
    profile: str | None,
    profile_path: list[str] | None,
    strictness: str | None,
    uses: list[str] | None,
    autodetect: bool,
    blending: bool,
    doc_warnings: bool | None,
    test_warnings: bool | None,
    member_warnings: bool | None,
    no_style_warnings: bool | None,
    full_pep8: bool | None,
    max_line_length: int | None,
    absolute_paths: bool,
    no_external_config: bool,
    pylint_config_file: str | None,
    show_profile: bool,
    ignore_paths: list[str] | None,
    ignore_patterns: list[str] | None,
) -> list[str]:
    args = ["prospector"]
    if not autodetect:
        args.append("--no-autodetect")
    if not blending:
        args.append("--no-blending")
    if doc_warnings:
        args.append("--doc-warnings")
    if test_warnings:
        args.append("--test-warnings")
    if member_warnings:
        args.append("--member-warnings")
    if no_style_warnings:
        args.append("--no-style-warnings")
    if full_pep8:
        args.append("--full-pep8")
    if max_line_length:
        args.extend(["--max-line-length", str(max_line_length)])
    if absolute_paths:
        args.append("--absolute-paths")
    if no_external_config:
        args.append("--no-external-config")
    if pylint_config_file:
        args.extend(["--pylint-config-file", pylint_config_file])
    if show_profile:
        args.append("--show-profile")
    if strictness:
        args.extend(["--strictness", strictness])
    if profile:
        args.extend(["--profile", profile])
    if profile_path:
        for path in profile_path:
            args.extend(["--profile-path", path])
    if uses:
        for use in uses:
            args.extend(["--uses", use])
    if tools:
        for tool in tools:
            args.extend(["--tool", tool])
    if with_tools:
        for tool in with_tools:
            args.extend(["--with-tool", tool])
    if without_tools:
        for tool in without_tools:
            args.extend(["--without-tool", tool])
    if ignore_paths:
        for path in ignore_paths:
            args.extend(["--ignore-paths", path])
    if ignore_patterns:
        for pattern in ignore_patterns:
            args.extend(["--ignore-patterns", pattern])
    args.extend(target_paths)
    return args


@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_VALIDATION, TaskType.REFACTOR)
def code_quality(
    root: str | Path | None = None,
    paths: list[str] | None = None,
    tools: list[str] | None = None,
    with_tools: list[str] | None = None,
    without_tools: list[str] | None = None,
    profile: str | None = None,
    profile_path: list[str] | None = None,
    strictness: str | None = None,
    uses: list[str] | None = None,
    autodetect: bool = True,
    blending: bool = True,
    doc_warnings: bool | None = None,
    test_warnings: bool | None = None,
    member_warnings: bool | None = None,
    no_style_warnings: bool | None = None,
    full_pep8: bool | None = None,
    max_line_length: int | None = None,
    absolute_paths: bool = False,
    no_external_config: bool = False,
    pylint_config_file: str | None = None,
    show_profile: bool = False,
    exclude_folders: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_messages: int | None = DEFAULT_MAX_RESULTS,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Run Prospector and return structured code-quality results.

    Args:
        root: Root directory to analyze (defaults to current working directory)
        paths: Specific paths to check (files or directories). If None, analyzes entire root
        tools: Only run these specific tools (e.g., ['pylint', 'pyflakes'])
        with_tools: Add these tools to the default set
        without_tools: Exclude these tools from the default set
        profile: Use a predefined Prospector profile (e.g., 'strictness_medium')
        profile_path: Paths to custom profile YAML files
        strictness: Set the strictness level ('veryhigh', 'high', 'medium', 'low', 'verylow')
        uses: Specify libraries/frameworks in use (e.g., ['django', 'celery'])
        autodetect: Automatically detect libraries and frameworks
        blending: Blend together settings from multiple tools
        doc_warnings: Report missing or malformed docstrings
        test_warnings: Report warnings in test files
        member_warnings: Report warnings about class/instance members
        no_style_warnings: Disable style-related warnings
        full_pep8: Enable all PEP8 checks (overrides some other settings)
        max_line_length: Maximum line length for style checks
        absolute_paths: Use absolute paths in output instead of relative
        no_external_config: Ignore project-level config files (.pylintrc, setup.cfg, etc.)
        pylint_config_file: Path to a specific pylint configuration file
        show_profile: Include profile information in output
        exclude_folders: Folder names to exclude from analysis
        exclude_patterns: File patterns to exclude (e.g., ['*_test.py'])
        max_messages: Maximum number of messages to return (0 for unlimited)
        allow_outside_root: Allow analyzing paths outside the root directory
        state: Agent state for configuration (internal)
    """
    root_path = resolve_root(root)
    config = get_config_from_state(state)

    try:
        from prospector.config import ProspectorConfig
        from prospector.run import Prospector
    except Exception as exc:
        return {
            "root": str(root_path),
            "files": 0,
            "summary": {},
            "message_count": 0,
            "messages": [],
            "truncated": False,
            "errors": [{"message": f"Failed to import prospector: {exc}"}],
        }

    target_paths: list[Path] = []
    if paths:
        for raw in paths:
            target = resolve_path(raw, root_path, allow_outside_root=allow_outside_root)
            target_paths.append(target)
    else:
        target_paths = [root_path]

    ignore_paths = sorted(set(get_exclude_dirs(config)) | set(exclude_folders or []))
    ignore_patterns = exclude_patterns[:] if exclude_patterns else []

    runs: list[dict] = []
    messages_out: list[dict] = []
    errors: list[dict] = []
    total_files = 0
    truncated = False

    def run_for_paths(run_paths: list[Path]) -> None:
        args = _build_prospector_args(
            [str(path) for path in run_paths],
            tools=tools,
            with_tools=with_tools,
            without_tools=without_tools,
            profile=profile,
            profile_path=profile_path,
            strictness=strictness,
            uses=uses,
            autodetect=autodetect,
            blending=blending,
            doc_warnings=doc_warnings,
            test_warnings=test_warnings,
            member_warnings=member_warnings,
            no_style_warnings=no_style_warnings,
            full_pep8=full_pep8,
            max_line_length=max_line_length,
            absolute_paths=absolute_paths,
            no_external_config=no_external_config,
            pylint_config_file=pylint_config_file,
            show_profile=show_profile,
            ignore_paths=ignore_paths,
            ignore_patterns=ignore_patterns,
        )
        try:
            with _temporary_argv(args):
                config = ProspectorConfig(workdir=root_path)
                prospector = Prospector(config)
                prospector.execute()
        except SystemExit as exc:
            errors.append({"message": f"Prospector exited with code {exc.code}"})
            return
        except Exception as exc:
            errors.append({"message": f"Prospector failed: {exc}"})
            return

        summary = prospector.get_summary() or {}
        runs.append({
            "paths": [str(path) for path in run_paths],
            "summary": _serialize_summary(summary),
        })

        for message in prospector.get_messages():
            messages_out.append(_serialize_message(message, root_path, absolute_paths))

    for path in target_paths:
        if path.is_dir():
            total_files += len(_iter_python_files(path, exclude_folders=exclude_folders, config=config))
        else:
            total_files += 1

    if len(target_paths) > 1 and any(path.is_dir() for path in target_paths):
        for path in target_paths:
            run_for_paths([path])
    else:
        run_for_paths(target_paths)

    counts_by_source: Counter[str] = Counter()
    counts_by_code: Counter[str] = Counter()
    for item in messages_out:
        counts_by_source[item["source"]] += 1
        counts_by_code[f"{item['source']}:{item['code']}"] += 1

    total_message_count = len(messages_out)
    if max_messages is not None and max_messages > 0 and len(messages_out) > max_messages:
        messages_out = messages_out[:max_messages]
        truncated = True

    summary = runs[-1]["summary"] if runs else {}
    return {
        "root": str(root_path),
        "files": total_files,
        "summary": summary,
        "runs": runs,
        "message_count": total_message_count,
        "messages_returned": len(messages_out),
        "messages": messages_out,
        "by_source": dict(counts_by_source.most_common()),
        "by_code": dict(counts_by_code.most_common()),
        "truncated": truncated,
        "errors": errors[:DEFAULT_MAX_RESULTS],
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    print(f"Analyzing project structure at: {project_root}")
    exclude_folders = ["challenges", "tests"]

    print("\n=== High-Level Module Dependency Graph ===")
    graph_data = python_dependency_graph(
        root=project_root,
        include_external=True,
        include_styling=True,
        exclude_folders=exclude_folders,
        max_nodes=0,
        max_edges=0,
    )
    print(f"Files: {graph_data['files']}")
    print(f"Modules: {graph_data['node_count']} (external: {len(graph_data['external_nodes'])})")
    print(f"Edges: {graph_data['edge_count']}")
    if graph_data["truncated"]["nodes"] or graph_data["truncated"]["edges"]:
        print(f"Truncated: nodes={graph_data['truncated']['nodes']}, edges={graph_data['truncated']['edges']}")

    out_counts: Counter[str] = Counter()
    in_counts: Counter[str] = Counter()
    for edge in graph_data["edges"]:
        out_counts[edge["from"]] += 1
        in_counts[edge["to"]] += 1

    print("\nTop outgoing dependencies:")
    for name, count in out_counts.most_common(10):
        print(f"  {name}: {count}")
    if not out_counts:
        print("  (none)")

    print("\nTop incoming dependencies:")
    for name, count in in_counts.most_common(10):
        print(f"  {name}: {count}")
    if not in_counts:
        print("  (none)")

    print("\nModules:")
    for name in graph_data["nodes"]:
        print(f"  {name}")

    print("\nExternal modules:")
    if graph_data["external_nodes"]:
        for name in graph_data["external_nodes"]:
            print(f"  {name}")
    else:
        print("  (none)")

    print("\nEdges:")
    for edge in graph_data["edges"]:
        print(f"  {edge['from']} -> {edge['to']}")

    if graph_data["errors"]:
        print("\nParse errors:")
        for err in graph_data["errors"]:
            print(f"  {err['file']}:{err['line']}:{err['column']} {err['message']}")

    print("\nDOT:")
    print(graph_data["dot"])

    print("\n=== Deep Dependency Graph (Classes & Functions) ===")
    deep_data = deep_python_dependency_graph(
        root=project_root,
        include_external=True,
        include_styling=True,
        exclude_folders=exclude_folders,
        max_nodes=0,
        max_edges=0,
    )
    print(f"Files: {deep_data['files']}")
    print(f"Nodes: {deep_data['node_count']} (external targets: {len(deep_data['external_nodes'])})")
    print(f"Edges: {deep_data['edge_count']}")
    if deep_data["truncated"]["nodes"] or deep_data["truncated"]["edges"]:
        print(f"Truncated: nodes={deep_data['truncated']['nodes']}, edges={deep_data['truncated']['edges']}")

    type_counts: Counter[str] = Counter()
    for node in deep_data["nodes"]:
        type_counts[node["type"]] += 1
    print("\nNode types:")
    for node_type, count in type_counts.most_common():
        print(f"  {node_type}: {count}")
    if not type_counts:
        print("  (none)")

    print("\nNodes:")
    for node in deep_data["nodes"]:
        parent = node.get("parent")
        if parent:
            print(f"  {node['type']}: {node['id']} (parent={parent})")
        else:
            print(f"  {node['type']}: {node['id']}")

    print("\nEdges:")
    for edge in deep_data["edges"]:
        edge_type = edge.get("type")
        if edge_type:
            print(f"  {edge['source']} -> {edge['target']} [{edge_type}]")
        else:
            print(f"  {edge['source']} -> {edge['target']}")

    print("\nExternal targets:")
    if deep_data["external_nodes"]:
        for name in deep_data["external_nodes"]:
            print(f"  {name}")
    else:
        print("  (none)")

    if deep_data["errors"]:
        print("\nParse errors:")
        for err in deep_data["errors"]:
            line = err.get("line", 0)
            message = err.get("message", "")
            print(f"  {err['file']}:{line} {message}")

    print("\nDOT:")
    print(deep_data["dot"])

    print("\n=== Unused Code (Vulture) ===")
    unused_data = python_unused_code(
        root=project_root,
        exclude_folders=exclude_folders,
    )
    print(f"Files: {unused_data['files']}")
    print(f"Min confidence: {unused_data['min_confidence']}")
    print(f"Unused items: {len(unused_data['unused_items'])}")

    unused_type_counts: Counter[str] = Counter()
    for item in unused_data["unused_items"]:
        unused_type_counts[item["type"]] += 1

    print("\nUnused items by type:")
    if unused_type_counts:
        for item_type, count in unused_type_counts.most_common():
            print(f"  {item_type}: {count}")
    else:
        print("  (none)")

    print("\nUnused items:")
    if unused_data["unused_items"]:
        for item in unused_data["unused_items"]:
            message = item.get("message") or ""
            if message:
                message = f" - {message}"
            print(
                f"  {item['type']}: {item['module']}:{item['line']} "
                f"{item['name']} ({item['confidence']}%){message}"
            )
    else:
        print("  (none)")

    if unused_data["errors"]:
        print("\nParse errors:")
        for err in unused_data["errors"]:
            line = err.get("line", 0)
            message = err.get("message", "")
            print(f"  {err['file']}:{line} {message}")

    print("\nNotes:")
    print("  Uses vulture; dynamic imports, getattr/registry lookups, and runtime usage may be missed.")

    print("\n=== Code Quality (Prospector) ===")
    quality_data = code_quality(
        root=project_root,
        exclude_folders=exclude_folders,
        max_messages=0,
    )
    print(f"Files: {quality_data['files']}")
    print(f"Messages: {quality_data['message_count']}")
    if quality_data["summary"]:
        summary = quality_data["summary"]
        tools = summary.get("tools", [])
        libraries = summary.get("libraries", [])
        strictness = summary.get("strictness", None)
        profiles = summary.get("profiles", "")
        if tools:
            print(f"Tools: {', '.join(tools)}")
        if libraries:
            print(f"Libraries: {', '.join(libraries)}")
        if strictness:
            print(f"Strictness: {strictness}")
        if profiles:
            print(f"Profiles: {profiles}")

    print("\nMessages by source:")
    if quality_data["by_source"]:
        for source, count in quality_data["by_source"].items():
            print(f"  {source}: {count}")
    else:
        print("  (none)")

    print("\nMessages:")
    if quality_data["messages"]:
        for message in quality_data["messages"]:
            path = message["path"] or message["absolute_path"] or "(unknown)"
            line = message["line"] or 0
            column = message["column"] or 0
            print(
                f"  {message['source']}:{message['code']} "
                f"{path}:{line}:{column} {message['message']}"
            )
    else:
        print("  (none)")

    if quality_data["errors"]:
        print("\nErrors:")
        for err in quality_data["errors"]:
            print(f"  {err.get('message', '')}")
