# Agentic logic

This document describes the planning and execution flow in Codur and how tools, agents, and state are coordinated.

## Graph overview

The main graph is defined in `codur/graph/main_graph.py`:

- pattern_plan -> llm_pre_plan -> llm_plan -> (tool | delegate | coding | explaining) -> execute -> review -> loop

## Planning phases

Phase 0: Pattern plan (textual, no LLM)
- Implemented in `codur/graph/planning/core.py` as `pattern_plan`.
- Uses quick pattern classification and the non-LLM tool detector.
- Can short-circuit with a direct response or tool calls.
- Tool detection from text is gated by `runtime.detect_tool_calls_from_text`.

Phase 1: LLM pre-plan (textual classification)
- Optional, enabled by `planning.use_llm_pre_plan`.
- Uses a small JSON classification prompt to identify task type.
- Only high-confidence trivial tasks are resolved here; most tasks continue to Phase 2.

Phase 2: LLM planner
- Full JSON planner that selects action: respond, tool, or delegate.
- Prompt is built with strategy-specific context from `codur/graph/planning/strategies/`.
- Uses the centralized tool registry to list available tools.
- Inserts file discovery and read_file steps when the task references code but lacks file context.
- When delegating to `agent:codur-coding`, it enforces file context by inserting read/list tools.

## Tool detection usage

Tool detection is still active in two places:

- Phase 0 pattern plan uses `codur/graph/tool_detection.py` to parse explicit tool calls in the user message (if `runtime.detect_tool_calls_from_text` is true).
- Execution uses the same detector to parse tool calls emitted by agents during tool loops. This is always on for agent execution.

## Tool registry and schema generation

Tools are centralized in `codur/tools/__init__.py` and exposed via `codur/tools/registry.py`.

- `list_tool_directory` and `get_tool_help` provide introspection.
- `codur/tools/schema_generator.py` turns tool signatures into JSON schemas.
- The coding agent builds a tool list for its system prompt via `list_tools_for_tasks`, using `TaskType` categories and fallback heuristics for unannotated tools.
- Full tool schemas from `get_function_schemas` are still provided to the LLM; the prompt list is for readability only.
- The coding agent prefers native tool calling with schemas; JSON fallback is used only when the provider does not support native tools.

## Tool execution

All tool execution is centralized in `codur/graph/tool_executor.py`.

- A single tool map binds tool names to their implementations.
- `AgentState` is wrapped and passed to tools so they can access config and message context.
- Paths are validated using `resolve_path` and `validate_file_access`, with secret and workspace guards.
- Test file overwrites are blocked unless explicitly requested.

## Tool injectors

Language-specific tool injectors live in `codur/graph/planning/injectors/`.

- Injectors add followup tools after read_file calls.
- Current injectors: Python and Markdown.
- Injection is used both in tool detection and when planning reads files for coding.
- Note: `inject_followup_tools` can coalesce multiple read_file calls into a `read_files` call. A `read_files` tool is not yet implemented in the tool executor.

## Agents

- `codur-coding` is a dedicated coding node with a tool-driven loop.
- It uses `create_and_invoke_with_tool_support` to pick native tool calling vs JSON fallback based on provider capabilities.
- It can retry with a configured fallback profile and loops tool execution up to 4 rounds (initial + 3 retries).
- `codur-explaining` is a dedicated explanation node.
- Other agents are configured via `codur.yaml` and loaded through `AgentRegistry`.
- Tools can call agents directly via `agent_call` and `retry_in_agent`.

## AgentState propagation

`AgentState` carries:
- Message history
- Tool calls and results
- Selected agent
- Iteration and LLM call counters
- Config object

Tool execution and agent invocation propagate this state so tools and agents can make consistent decisions.

## Review and verification

The review node orchestrates verification for fix/debug tasks. It operates in two layers:

### Review Node (`codur/graph/execution/review.py`)
- **Task detection**: Identifies fix/debug tasks by keywords (fix, bug, error, debug, implement, etc.)
- **Verification orchestration**: Calls `verification_agent_node` to determine if implementation satisfies requirements
- **Result routing**:
  - Success → return END
  - Repeated error (same error 2+ times) → return END (agent is stuck)
  - Verification failure → attempt local repair or route for retry
- **Retry logic**:
  - Up to iteration limit: Route failed verification back to coding agent for retry
  - After 3 failed attempts: Route back to planning for fresh approach
- **Local repair fallback**: Attempts 7 common mutation patterns (range boundaries, comparison operators, etc.) as last resort

### Verification Agent (`codur/graph/execution/verification_agent.py`)
- **Dynamic strategy inference**: Analyzes original request and project structure to choose verification approach:
  - **Test-based**: Finds and runs tests (test_*.py, *_test.py) via pytest when tests exist
  - **Execution-based**: Discovers entry points, runs them, captures output, optionally compares against expected outputs
  - **Static analysis**: Validates syntax, checks code quality
  - **Hybrid**: Combines multiple strategies
- **Context-driven**: No hardcoded file names (main.py, app.py, expected.txt) — infers from actual project structure
- **Tool-driven**: Uses tools like `discover_entry_points`, `run_python_file`, `run_pytest`, `list_files`, etc.
- **Structured results**: Returns pass/fail decision with reasoning, expected vs actual output, and specific suggestions for fixing failures

### Error Context
- Failed verifications create structured error messages with:
  - Expected output (if applicable)
  - Actual output or error message (truncated for readability)
  - Current implementation context
  - Verification agent's suggestions for fixes
- Error hashes track repeated failures to detect when agent is stuck in a loop

## LLM configuration

Groq is the default planner provider, but the system is configuration driven:

- LLM profiles and providers are defined in `codur.yaml`.
- Agents can be LLM profiles or tool-based (Codex, Claude Code, MCP).
- Fallback profiles are supported for planning.

## LLM invocation and limits

- Each `invoke_llm` call increments `llm_calls` in `AgentState` and enforces `max_llm_calls` from state or config.
- `model_agent_instructions` injects system messages when `invoked_by` matches configured patterns (including `"all"`).

## Observability

Observability and tracing hooks are deprecated. Current visibility is provided by Rich logging and the TUI debug panel.

## Desired tool metadata

Tools currently lack metadata that indicates when they are appropriate. Adding registry-level metadata or tags is a desired improvement.

## See also

- `codur/graph/execution/README.md` for execution, delegation, and review node architecture
- `CODING.md` for the coding agent tool loop specifics
- `codur/tools/README.md` for tool registry details and authoring guidance
- `codur/utils/README.md` for preferred shared utilities
- `codur/graph/planning/injectors/README.md` for language-specific tool injectors
- `codur/graph/planning/strategies/README.md` for textual planning strategies
