# Codur

Codur is an autonomous coding agent orchestrator built on LangGraph. It routes tasks to LLMs and tools, runs those tools safely inside a workspace, and retries with verification for code-fix style tasks.

## Requirements

- Python 3.10+
- Provider API keys as needed (Groq is the default)
- Optional CLI tools used by some tools: rg, git, pandoc

## Install

```bash
python -m pip install -e .
```

## Configure

- Default config: `codur.yaml` in the repo root
- Optional user config: `~/.codur/config.yaml`
- Override config path with `--config path/to/config.yaml`

Set provider keys in your shell:

```bash
export GROQ_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

You can also update the planner model via the CLI:

```bash
codur configure --provider groq --model qwen/qwen3-32b
```

## Run (CLI)

```bash
codur -c "Fix the bug in main.py"
```

```bash
codur run "Refactor the auth module" --verbose
```

Use `--raw` if you only want the final response text.

## Run (TUI)

```bash
python -m codur.tui
```

The TUI uses the same configuration and supports live guidance.

## Validation and challenges

The `challenges/` directory is used for regression testing. See `challenges/README.md` for structure and how to run the harness.

## Docs

- `CLAUDE.md` - Runtime behavior and contributor guardrails
- `AGENTIC_LOGIC.md` - Planning and execution flow details
- `codur/tools/README.md` - Tool registry and authoring
- `codur/graph/nodes/planning/injectors/README.md` - Tool injectors
- `codur/graph/nodes/planning/strategies/README.md` - Planning strategies

## Contributing

- Prefer centralized utilities in `codur/utils` for paths, validation, and LLM calls.
- Avoid challenge-only fixes; changes should be generalizable.
