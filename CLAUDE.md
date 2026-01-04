# CLAUDE.md

This document explains how Codur works at a high level and how to make changes safely. It is intentionally brief and points to deeper references.

## How the application works

- The LangGraph workflow runs: pattern_plan -> llm_classification -> llm_plan -> (tool | delegate | coding | explaining) -> execute -> review -> loop.
- Planning is JSON-based and uses Groq by default, but all providers and profiles are configuration driven.
- The coding agent prefers native tool calling with schemas and falls back to JSON tool calls when native tooling is unavailable.
- Tools execute through a centralized tool executor and receive AgentState for context.

## Configuration and entrypoints

- CLI: `codur` (Typer) in `codur/cli.py`
- TUI: `python -m codur.tui`
- Primary config: `codur.yaml`

## Guardrails

- **Do not hardcode challenge-specific patterns**: Never assume `expected.txt`, or other challenge-specific file names in tools or agents. The review node already handles challenge validation. Any new tool must work generically across any Python project, not just the test fixtures.
- Prefer centralized utilities in `codur/utils` over local ad-hoc helpers.
- Keep all tool execution logic in `codur/graph/tool_executor.py`.
- Do not add code that only satisfies the `challenges/` fixtures; changes must generalize.
- Observability/tracing is deprecated; rely on Rich logging and the TUI debug view.
- Tools, strategies, and agents should use TaskType annotations when appropriate.
- Local imports should be avoided

## Further reading

- `AGENTIC_LOGIC.md` for the full planner and execution flow.
- `CODING.md` for the coding agent tool loop and guardrails.
- `codur/utils/README.md` for preferred shared utilities and agent helpers.
- `codur/graph/execution/README.md` for execution, delegation, and review nodes.
- `codur/tools/README.md` for tool registry and authoring guidance.
- `codur/graph/planning/injectors/README.md` for language-specific tool injectors.
- `codur/graph/planning/strategies/README.md` for textual planning strategies.
- `challenges/README.md` for the validation harness.
