# Codur Utilities

This directory contains shared, workspace-safe helpers that agents and tools should prefer over ad-hoc local logic. If you are about to re-implement behavior that already exists here, stop and reuse the utility instead.

## Rules of use

- Prefer these utilities over local helpers or inline logic.
- Keep path and permission checks centralized (do not bypass `path_utils` or `validation`).
- Avoid local imports; use these shared modules directly.
- Do not hardcode challenge-specific patterns or filenames.

## Utility catalog (preferred helpers)

### Filesystem safety and paths

- `codur/utils/path_utils.py`
  - `resolve_root`, `resolve_path`, `set_default_root`
  - Use for workspace-aware path resolution and root enforcement.
- `codur/utils/validation.py`
  - `validate_file_access`, `validate_within_workspace`, `require_*`
  - Use for file existence checks, workspace boundaries, and tool permission gating.
- `codur/utils/ignore_utils.py`
  - `guard_secret_read`, `get_exclude_dirs`, `load_gitignore`, `is_gitignored`
  - Use for gitignore/secret guards and hidden file policy.
- `codur/utils/path_extraction.py`
  - `extract_path_from_message`, `extract_file_paths`, `find_workspace_match`
  - Use for parsing user text into candidate file paths.

### LLM invocation and retries

- `codur/utils/llm_calls.py`
  - `invoke_llm` and LLM call limit tracking with `LLMCallLimitExceeded`.
  - Use for counting and limiting LLM calls and injecting agent instructions.
- `codur/utils/llm_helpers.py`
  - `create_and_invoke`, `create_and_invoke_with_tool_support`
  - Use for provider-aware LLM creation and tool calling (native or JSON fallback).
- `codur/utils/retry.py`
  - `retry_with_backoff`, `LLMRetryStrategy`
  - Use for network/LLM retry logic with backoff and fallback profiles.

### Tool call handling

- `codur/utils/tool_helpers.py`
  - `extract_tool_info`
  - Use for validating tool call structure and supported tool lists.
- `codur/utils/tool_response_handler.py`
  - `deserialize_tool_calls`, `extract_tool_calls_from_json_text`
  - Use for normalizing tool calls across native and JSON fallback responses.

### Text and output helpers

- `codur/utils/text_helpers.py`
  - `truncate_lines`, `truncate_chars`, `truncate_text`, `smart_truncate`
  - Use for safe output truncation and summarization.

### Git utilities

- `codur/utils/git.py`
  - `get_diff_for_path`
  - Use for diff generation without shelling out to git.

### Configuration access

- `codur/utils/config_helpers.py`
  - `get_or_default`, `require_config`, `get_max_iterations`, `get_cli_timeout`
  - Use for safe, defaulted config lookups and common runtime settings.

## Common patterns to follow

- Always resolve paths via `resolve_path` and validate them with `validate_file_access`.
- Always guard secret and ignored paths via `ignore_utils` (do not re-implement globs or gitignore checks).
- Use `create_and_invoke_with_tool_support` for tool-enabled LLM calls; do not re-implement tool fallback logic.
- Use `tool_response_handler` and `tool_helpers` to parse and validate tool calls.
- Use `retry` helpers instead of custom retry loops.

## When to add a new utility

Add a helper here when logic is reused across tools, agents, or graph nodes. Keep utilities general and workspace-safe, and update this README when new helpers are added.
