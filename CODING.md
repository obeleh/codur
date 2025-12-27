# CODING.md

This document describes the dedicated coding agent (`codur-coding`) and how it uses tools to implement changes.

## Role

`codur-coding` is a specialized node for code implementation tasks. It is invoked when planning decides a task requires code changes and sufficient file context is available (or after planning inserts read steps).

## Entry points

- Coding node: `codur/graph/nodes/coding.py`
- Tool executor: `codur/graph/nodes/tool_executor.py`
- Tool schemas: `codur/tools/schema_generator.py`

## Execution flow

1. Build a structured prompt from `AgentState` messages.
2. Load tool schemas from the centralized tool registry.
3. Invoke the LLM with native tool calling when supported.
4. Parse tool calls (native API or JSON fallback).
5. Execute tool calls via the centralized tool executor.
6. Feed tool results back into the LLM and repeat until no more tool calls.

The tool loop is capped (default 5 iterations) to avoid runaway calls.

## Tool calling behavior

- Preferred path: native tool calling with schemas.
- Fallback path: JSON tool calls in the response text when the provider does not support native tools.
- Tool calls are executed by `execute_tool_calls`, which handles path safety and state propagation.
- The coding node avoids re-reading files by explicitly tracking and warning on read_file repeats.

## Automatic validation

The tool executor automatically injects syntax validation after Python code modifications:
- When code modification tools (`write_file`, `replace_function`, `replace_class`, `replace_method`, `replace_file_content`, `inject_function`) target Python files, a `validate_python_syntax` call is automatically injected
- This validates the new code before changes are applied
- Additionally, these tools validate syntax internally and report errors if invalid code is provided

## Guardrails

- Only tools exposed via the tool registry are allowed.
- Tools must include required parameters (notably `path` for code changes).
- Test file overwrites are blocked unless explicitly requested.
- The system prompt warns against inventing tools and encourages targeted changes.
- Python code modifications are automatically validated for syntax errors.

## Configuration

- Planner and coding LLMs use profiles from `codur.yaml`.
- Default provider is Groq, but profiles are fully configurable.
- Fallback model for the coding agent is `agents.preferences.fallback_model`.

## Known limitations

- Tool metadata for “appropriateness” is desired but not implemented.
- `inject_followup_tools` may coalesce read calls into `read_files`, which is not yet implemented as a tool.

## Related docs

- `AGENTIC_LOGIC.md` for the full planning and execution flow
- `codur/tools/README.md` for tool registry and tool authoring
- `codur/graph/nodes/planning/injectors/README.md` for tool injection
