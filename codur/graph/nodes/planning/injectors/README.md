# Tool injectors

Tool injectors are a centralized system for injecting language-specific tools after `read_file` calls. They replace scattered file-extension checks across the codebase.

## Where injection happens

Injection is used in two places:

- `codur/graph/nodes/tool_detection.py` when the non-LLM detector finds tool calls in user text
- `codur/graph/nodes/planning/core.py` when planning reads files before delegating to the coding agent

## Registry and protocol

- Protocol: `codur/graph/nodes/planning/injectors/base.py`
- Registry: `codur/graph/nodes/planning/injectors/registry.py`
- Re-exports: `codur/graph/nodes/planning/injectors/__init__.py`

Each injector implements:

- `extensions` (file suffixes)
- `name` (display name)
- `get_followup_tools(paths)`
- `get_planning_tools()`
- `get_example_tool_calls(path)`

## inject_followup_tools behavior

`inject_followup_tools`:

- Collects all `read_file` tool calls and groups them by injector
- Adds followup tools for each injector
- De-duplicates identical tool calls

Note on multi-file reads:

- When multiple `read_file` calls are present, the injector registry can coalesce them into a single `read_files` call.
- There is no `read_files` tool in the tool executor yet, so multi-file reads should remain as individual `read_file` calls until that tool is implemented.

## Supported injectors

Python (`.py`, `.pyi`)
- Followup tools: `python_ast_dependencies` or `python_ast_dependencies_multifile`
- Planning tools: `python_ast_outline`, `python_ast_graph`, `python_dependency_graph`
- Automatic syntax validation (tool executor): When code modification tools (`write_file`, `replace_function`, `replace_class`, `replace_method`, `replace_file_content`, `inject_function`) are called on Python files, the tool executor automatically injects a `validate_python_syntax` call to check the new code for syntax errors before applying changes.
- Built-in validation: Code modification tools also validate Python syntax internally before making changes and report errors if syntax is invalid. The `validate_python_syntax` tool is available for explicit validation when needed.
- Runtime validation: The `run_python_file` tool allows the LLM to execute Python files and validate behavior during the coding phase.

Markdown (`.md`, `.markdown`)
- Followup tools: `markdown_outline`
- Planning tools: `markdown_outline`, `markdown_extract_sections`, `markdown_extract_tables`

## Adding a new injector

1. Create a new file in `codur/graph/nodes/planning/injectors/`.
2. Implement the injector protocol.
3. Register it in `codur/graph/nodes/planning/injectors/registry.py`.
4. Add any new tools in `codur/tools/` and export them in `codur/tools/__init__.py`.
5. Add tests under `tests/py_only/graph/nodes/planning/injectors/`.

## Testing

```bash
pytest tests/py_only/graph/nodes/planning/injectors/
```
