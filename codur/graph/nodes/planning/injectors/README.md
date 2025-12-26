# Tool Injectors

Tool Injectors are a centralized system for automatically injecting language-specific tools when reading files of different types. This system eliminates the need for scattered file extension checks throughout the codebase and makes it easy to add support for new file types.

## Overview

When a task involves reading a file (e.g., `read_file` tool call), the injector system automatically adds appropriate followup tools based on the file's extension:

- **Python files** (`.py`, `.pyi`) → Automatically inject AST analysis tools
- **Markdown files** (`.md`, `.markdown`) → Automatically inject document outline tools
- **Other languages** → Easy to add (see "Adding a New Language" below)

This happens transparently in two places:
1. **Tool Detection** (`tool_detection.py`) - When detecting tools from user messages
2. **Planning Core** (`planning/core.py`) - When gathering context before delegating to agents

## Architecture

### The ToolInjector Protocol

Each language has a **Tool Injector** that implements a simple protocol:

```python
class ToolInjector(Protocol):
    @property
    def extensions(self) -> FrozenSet[str]:
        """File extensions this injector handles (e.g., {'.py', '.pyi'})."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name (e.g., 'Python', 'Markdown')."""
        ...

    def get_followup_tools(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Get additional tools to inject when reading these file paths.

        Args:
            paths: List of file paths being read

        Returns:
            List of tool calls to inject
        """
        ...

    def get_planning_tools(self) -> List[str]:
        """Get tool names to suggest in planning prompts."""
        ...

    def get_example_tool_calls(self, example_path: str) -> List[Dict[str, Any]]:
        """Get example tool calls for planning prompt examples."""
        ...
```

### The Registry

All injectors are registered in `__init__.py`:

```python
from .python import PythonToolInjector
from .markdown import MarkdownToolInjector

_INJECTORS: List[ToolInjector] = [
    PythonToolInjector(),
    MarkdownToolInjector(),
]

# Extension -> Injector mapping (computed once)
_EXTENSION_MAP: Dict[str, ToolInjector] = {}
for injector in _INJECTORS:
    for ext in injector.extensions:
        _EXTENSION_MAP[ext.lower()] = injector
```

### Core Functions

**`get_injector_for_file(path: str) -> Optional[ToolInjector]`**
- Returns the appropriate injector for a file path
- Uses file extension to find the right injector
- Returns `None` if no injector exists for that file type

**`inject_followup_tools(tool_calls: List[Dict]) -> List[Dict]`**
- Takes a list of tool calls
- Finds all `read_file` calls
- Groups them by injector type
- Adds appropriate followup tools
- Returns the extended tool call list

## Currently Supported Languages

### Python (`.py`, `.pyi`)

**Injector**: `PythonToolInjector` (`injectors/python.py`)

**Followup Tools**:
- Single file: `python_ast_dependencies`
- Multiple files: `python_ast_dependencies_multifile`

**Planning Tools**:
- `python_ast_outline`
- `python_ast_graph`
- `python_dependency_graph`
- `python_ast_dependencies`
- `python_ast_dependencies_multifile`

**Example**:
```python
# Input tool call
[{"tool": "read_file", "args": {"path": "main.py"}}]

# After injection
[
    {"tool": "read_file", "args": {"path": "main.py"}},
    {"tool": "python_ast_dependencies", "args": {"path": "main.py"}}
]
```

### Markdown (`.md`, `.markdown`)

**Injector**: `MarkdownToolInjector` (`injectors/markdown.py`)

**Followup Tools**:
- Each file: `markdown_outline`

**Planning Tools**:
- `markdown_outline`
- `markdown_extract_sections`
- `markdown_extract_tables`

**Example**:
```python
# Input tool call
[{"tool": "read_file", "args": {"path": "README.md"}}]

# After injection
[
    {"tool": "read_file", "args": {"path": "README.md"}},
    {"tool": "markdown_outline", "args": {"path": "README.md"}}
]
```

## Adding a New Language

To add support for a new file type:

### 1. Create the Injector Class

Create a new file in `codur/graph/nodes/planning/injectors/` (e.g., `javascript.py`):

```python
"""JavaScript tool injector."""

from typing import FrozenSet, List, Dict, Any


class JavaScriptToolInjector:
    """Tool injector for JavaScript source files."""

    @property
    def extensions(self) -> FrozenSet[str]:
        return frozenset({".js", ".jsx", ".mjs"})

    @property
    def name(self) -> str:
        return "JavaScript"

    def get_followup_tools(self, paths: List[str]) -> List[Dict[str, Any]]:
        """Inject ESLint analysis or AST parsing for JavaScript files."""
        return [
            {"tool": "eslint_check", "args": {"paths": paths}}
        ]

    def get_planning_tools(self) -> List[str]:
        """Tools to suggest in planning prompts."""
        return [
            "eslint_check",
            "javascript_ast_parse",
            "npm_dependencies",
        ]

    def get_example_tool_calls(self, example_path: str) -> List[Dict[str, Any]]:
        """Example showing read_file + eslint."""
        return [
            {"tool": "read_file", "args": {"path": example_path}},
            {"tool": "eslint_check", "args": {"paths": [example_path]}}
        ]
```

### 2. Register the Injector

Update `codur/graph/nodes/planning/injectors/__init__.py`:

```python
from .python import PythonToolInjector
from .markdown import MarkdownToolInjector
from .javascript import JavaScriptToolInjector  # Add this

_INJECTORS: List[ToolInjector] = [
    PythonToolInjector(),
    MarkdownToolInjector(),
    JavaScriptToolInjector(),  # Add this
]
```

### 3. (Optional) Create Language-Specific Tools

If your injector references new tools (like `eslint_check` above), create them in `codur/tools/`:

```python
# codur/tools/javascript.py
def eslint_check(paths: List[str]) -> str:
    """Run ESLint on JavaScript files."""
    # Implementation here
    pass
```

And export them in `codur/tools/__init__.py`:

```python
from codur.tools.javascript import eslint_check

__all__ = [
    # ... existing tools ...
    "eslint_check",
]
```

### 4. Write Tests

Create `tests/py_only/graph/nodes/planning/injectors/test_javascript_injector.py`:

```python
"""Tests for JavaScript tool injector."""

import pytest
from codur.graph.nodes.planning.injectors.javascript import JavaScriptToolInjector


class TestJavaScriptToolInjector:
    @pytest.fixture
    def injector(self):
        return JavaScriptToolInjector()

    def test_extensions(self, injector):
        assert ".js" in injector.extensions
        assert ".jsx" in injector.extensions

    def test_get_followup_tools(self, injector):
        paths = ["app.js"]
        tools = injector.get_followup_tools(paths)

        assert len(tools) == 1
        assert tools[0]["tool"] == "eslint_check"
```

That's it! Your new language is now fully integrated.

## How It Works

### Example Flow

1. **User Message**: `"Fix the bug in main.py"`

2. **Tool Detection** (`tool_detection.py`):
   - Detects: `[{"tool": "read_file", "args": {"path": "main.py"}}]`
   - Calls: `inject_followup_tools(tools)`
   - Registry finds: `PythonToolInjector` for `.py` extension
   - Adds: `{"tool": "python_ast_dependencies", "args": {"path": "main.py"}}`
   - Returns: Both tools

3. **Planning** (`planning/core.py`):
   - When delegating to `agent:codur-coding` without file context
   - Creates read_file calls for discovered files
   - Calls: `inject_followup_tools(tool_calls)`
   - Same injection process happens

4. **Result**:
   - Both the file contents AND AST analysis are available
   - Agent has full context to work with

### Multi-Language Example

```python
# Input: Reading both Python and Markdown files
tool_calls = [
    {"tool": "read_file", "args": {"path": "main.py"}},
    {"tool": "read_file", "args": {"path": "README.md"}}
]

# After inject_followup_tools()
[
    {"tool": "read_file", "args": {"path": "main.py"}},
    {"tool": "read_file", "args": {"path": "README.md"}},
    {"tool": "python_ast_dependencies", "args": {"path": "main.py"}},
    {"tool": "markdown_outline", "args": {"path": "README.md"}}
]
```

## Design Decisions

### Why Protocol-Based?

- **Duck typing**: No inheritance required
- **Easy testing**: Can mock with simple classes
- **Clear contract**: Protocol defines the interface explicitly
- **Extensible**: Add new injectors without modifying core files

### Why Code-Only Configuration?

- **Simplicity**: No YAML parsing, no schema validation
- **Type safety**: Python type checker validates everything
- **Discoverability**: Easy to find all injectors via imports
- **Flexibility**: Can use logic, not just static data

### Why Per-Extension Mapping?

- **Performance**: O(1) lookup by extension
- **Clarity**: One injector per file type
- **Flexibility**: Multiple extensions per injector (`.py` and `.pyi`)

## Benefits

1. **Centralization**: All language-specific logic in one place
2. **Consistency**: Same detection logic everywhere
3. **Maintainability**: Change behavior in one place
4. **Extensibility**: Add new languages by creating one file
5. **Testability**: Test injectors in isolation
6. **Discoverability**: Clear registry of supported languages
7. **Documentation**: Single source of truth for tool injection

## Migration from Old System

The old system used scattered `.endswith(".py")` checks in multiple files:

- `tool_detection.py` - Manual AST injection
- `planning/core.py` - Hardcoded Python file filtering
- Strategy files - Duplicate extension checks

This has been fully replaced by the injector system. The old `detect_followup_tools()` function has been removed.

## Future Enhancements

Potential languages to add:

- **TypeScript/JavaScript** (`.ts`, `.tsx`, `.js`, `.jsx`) - ESLint, AST parsing
- **Go** (`.go`) - `go vet`, imports analysis
- **Rust** (`.rs`) - `cargo clippy`, module structure
- **Java** (`.java`) - Dependency analysis, package structure
- **C/C++** (`.c`, `.cpp`, `.h`, `.hpp`) - Include analysis, compilation checks

Potential tool enhancements:

- **Markdown**: Frontmatter parsing, link validation, section replacement
- **Python**: Import graph visualization, coverage analysis, type checking
- **Configuration**: Auto-detect and parse config file formats

## Testing

All injectors have comprehensive test coverage:

- `tests/py_only/graph/nodes/planning/injectors/test_python_injector.py`
- `tests/py_only/graph/nodes/planning/injectors/test_markdown_injector.py`
- `tests/py_only/graph/nodes/planning/injectors/test_registry.py`

Run tests:
```bash
pytest tests/py_only/graph/nodes/planning/injectors/
```

## Related Documentation

- [Tool Detection](../../tool_detection.py) - How tools are detected from user messages
- [Planning Core](../core.py) - How planning decides what tools to use
- [Planning Strategies](../strategies/) - Task-specific planning strategies
- [Markdown Tools](../../../../tools/markdown.py) - Markdown-specific tool implementations
- [Python AST Tools](../../../../tools/python_ast.py) - Python AST analysis tools

---

**Last Updated**: 2025-12-26
**Status**: Production-ready
**Maintainers**: Codur Core Team
