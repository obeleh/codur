# Codur tools

Tools are plain Python functions. They are discovered via a centralized registry, turned into schemas for tool calling, and executed through a single tool executor.

This document also covers TaskType scoping and guidance for authoring new tools.

## How tools are discovered

- Export tools in `codur/tools/__init__.py` and add them to `__all__`.
- `codur/tools/registry.py` reads `__all__` to build the tool directory.
- `codur/tools/schema_generator.py` converts signatures into JSON schemas for tool calling.

## How tools execute

- `codur/graph/tool_executor.py` owns the execution map.
- Tool execution is centralized so path safety and state propagation are consistent.
- Tools receive `root`, `allow_outside_root`, and `state` so they can respect workspace boundaries and config.
- Path safety uses `resolve_path` and `validate_file_access`, plus gitignore and secret guards.

## Add a tool

1. Create a module in `codur/tools/` with the tool function.
2. Use centralized helpers like `resolve_path`, `resolve_root`, and `validate_file_access`.
3. Export the tool in `codur/tools/__init__.py` and add it to `__all__`.
4. Register the tool in the executor map inside `codur/graph/tool_executor.py`.
5. Keep the first docstring line short; it is used for summaries.
6. Return JSON-serializable output and keep responses small.
7. Add tests when the tool affects core behavior.

## Tool schemas

The coding agent uses tool schemas when the provider supports native tool calling. If native tool calling is not supported, it falls back to JSON tool calls that are parsed from the response text.

## Task-specific tools (TaskType)

Tools can be annotated with `TaskType` to signal when they are appropriate. The enum lives in `codur/constants.py`.

Example:

```python
from codur.constants import TaskType
from codur.tools.tool_annotations import tool_scenarios

@tool_scenarios(TaskType.CODE_FIX, TaskType.CODE_GENERATION)
def replace_function(...):
    ...
```

Filter tools by scenario:

```python
from codur.constants import TaskType
from codur.tools.registry import list_tools_for_tasks

tools = list_tools_for_tasks([TaskType.CODE_GENERATION])
```

The coding agent uses TaskType categories to build a readable tool list for its system prompt. Unannotated tools may still be included as a fallback, but explicit TaskType annotations are preferred.

## Tool execution context

Tools can declare how the executor should inject context (filesystem, search, or config):

```python
from codur.tools.tool_annotations import ToolContext, tool_contexts

@tool_contexts(ToolContext.FILESYSTEM)
def read_file(...):
    ...
```

Use `ToolContext.FILESYSTEM` for tools that need `root` and `allow_outside_root`, `ToolContext.SEARCH` for tools that only need `root`, and `ToolContext.CONFIG` for tools that require `config` injection.

## Usage examples

Run pytest with optional filters:

```python
run_pytest(paths=["tests"], keyword="api", markers="slow")
```

## Authoring guidance for LLMs

When adding a new tool, think in terms of safety, generality, and clarity:

1. **Scope and generality**
   - The tool must work across arbitrary Python projects.
   - Do not hardcode challenge-specific filenames or behavior.
   - Prefer centralized helpers in `codur/utils` over ad-hoc local logic.

2. **Registry + executor wiring**
   - Add the tool to `codur/tools/__init__.py` and `__all__`.
   - Ensure `codur/graph/tool_executor.py` includes it in the execution map.
   - Add TaskType annotations so it is discoverable for the correct tasks.

3. **Schema and parameters**
   - Keep parameters minimal, explicit, and JSON-serializable.
   - Any file-modification tool should accept a `path` parameter.
   - Be precise in docstrings: they become part of the tool schema.

4. **Validation and guardrails**
   - Respect workspace and secret path guards via existing utilities.
   - Prefer read or preview operations when possible; avoid destructive actions.
   - For Python edits, consider whether `validate_python_syntax` should be used or injected.

5. **State and context**
   - Tools receive `AgentState` via the executor for consistent context.
   - Avoid local imports; use shared modules from `codur/utils`.
   - Emit helpful error messages to guide retries and debugging.

6. **Testing and documentation**
   - Add tests for tools that change core behavior.
   - Update related docs (`AGENTIC_LOGIC.md`, `CODING.md`) if behavior changes.

## Example

```python
from __future__ import annotations

from pathlib import Path
from codur.graph.state import AgentState
from codur.utils.path_utils import resolve_path
from codur.utils.validation import validate_file_access
from codur.utils.ignore_utils import get_config_from_state


def example_tool(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """One-line summary shown in tool listings."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    config = get_config_from_state(state)
    validate_file_access(target, Path(root) if root else Path.cwd(), config)
    return {"path": str(target)}
```

## Notes

- Tools can call agents through `agent_call` or `retry_in_agent`.
- Prefer centralized utilities over local ad-hoc helpers.

## See also

- `AGENTIC_LOGIC.md` for the full planning and execution flow
- `CODING.md` for how the coding agent uses tools
