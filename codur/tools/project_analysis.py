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


def _iter_python_files(root: Path, exclude_folders: list[str] | None = None) -> list[Path]:
    files: list[Path] = []
    default_excludes = set(EXCLUDE_DIRS)
    
    # Pre-process exclude folders for platform compatibility
    norm_excludes = []
    if exclude_folders:
        norm_excludes = [ex.replace("/", os.sep) for ex in exclude_folders]

    for dirpath, dirnames, filenames in os.walk(root):
        # 1. Standard exclusions
        dirnames[:] = [d for d in dirnames if d not in default_excludes]
        
        # 2. Custom exclusions (path-based)
        if norm_excludes:
            allowed_dirs = []
            for d in dirnames:
                full_path = Path(dirpath) / d
                try:
                    rel_path = full_path.relative_to(root)
                    s_rel = str(rel_path)
                    
                    should_exclude = False
                    for ex in norm_excludes:
                        if s_rel == ex or s_rel.startswith(ex + os.sep):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        allowed_dirs.append(d)
                except ValueError:
                    # Should not happen if traversing inside root, but strictly safe
                    allowed_dirs.append(d)
            dirnames[:] = allowed_dirs

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


def _is_excluded_module(name: str, excludes: list[str] | None) -> bool:
    if not excludes:
        return False
    for ex in excludes:
        if name == ex or name.startswith(ex + "."):
            return True
    return False


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
    """
    root_path = resolve_root(root)
    file_paths: list[Path] = []

    if paths:
        for raw in paths:
            target = resolve_path(raw, root_path, allow_outside_root=allow_outside_root)
            if target.is_dir():
                file_paths.extend(_iter_python_files(target, exclude_folders=exclude_folders))
            elif target.is_file() and target.suffix == ".py":
                file_paths.append(target)
    else:
        file_paths = _iter_python_files(root_path, exclude_folders=exclude_folders)

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
    """
    root_path = resolve_root(root)
    file_paths: list[Path] = []

    if paths:
        for raw in paths:
            target = resolve_path(raw, root_path, allow_outside_root=allow_outside_root)
            if target.is_dir():
                file_paths.extend(_iter_python_files(target, exclude_folders=exclude_folders))
            elif target.is_file() and target.suffix == ".py":
                file_paths.append(target)
    else:
        file_paths = _iter_python_files(root_path, exclude_folders=exclude_folders)

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
    if max_nodes and len(all_nodes) > max_nodes:
        all_nodes = all_nodes[:max_nodes]
    
    node_ids = {n["id"] for n in all_nodes}
    
    # Convert set back to list of dicts and filter
    all_edges = []
    for s, t, ty in sorted(all_edges_set):
        # Exclude edges where target is an excluded module
        if exclude_modules and _is_excluded_module(t, exclude_modules):
            continue
        all_edges.append({"source": s, "target": t, "type": ty})
    
    filtered_edges = [e for e in all_edges if e["source"] in node_ids]
    if max_edges and len(filtered_edges) > max_edges:
        filtered_edges = filtered_edges[:max_edges]


if __name__ == "__main__":
    # print the project structure of this project
    project_root = Path(__file__).resolve().parents[2]
    print(f"Analyzing project structure at: {project_root}")

    print("\n=== High-Level Module Dependency Graph ===")
    graph_data = python_dependency_graph(root=project_root)
    print(f"Files: {graph_data['files']}, Nodes: {graph_data['node_count']}, Edges: {graph_data['edge_count']}")
    
    print("\n=== Deep Dependency Graph (Classes & Functions) ===")
    deep_data = deep_python_dependency_graph(root=project_root, include_external=True)
    print(f"Nodes: {deep_data['node_count']}, Edges: {deep_data['edge_count']}")
    print(f"Errors: {len(deep_data['errors'])}")
    
    print(deep_data['dot'])