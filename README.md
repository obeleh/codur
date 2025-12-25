# Codur - Autonomous Coding Agent Orchestrator

**Status:** 🚧 Experimental / In Development

A LangGraph-based agent orchestrator that analyzes coding tasks and delegates them to specialized agents (Claude Code, Codex, Ollama) with integrated tools and MCP servers.

## Core Capabilities

Codur orchestrates coding tasks through:
- **Three-phase planning system** - Fast pattern matching → optional pre-classification → full LLM planning
- **Multiple agents** - Claude Code (primary), Codex, Ollama (local), with intelligent routing
- **40+ integrated tools** - File operations, code analysis, git, structured data, web search, MCP tools
- **Challenge-based learning** - Built-in test framework for validating agent performance
- **YAML configuration** - Flexible configuration for agents, LLMs, MCP servers, and runtime settings

## Architecture

```
┌────────────────────────────────────────────────────────┐
│               Codur CLI / TUI                          │
│           (Command-line or Terminal UI)               │
└────────────────────┬─────────────────────────────────┘
                     │
      ┌──────────────▼──────────────┐
      │    LangGraph Orchestrator   │
      │  (3-phase planning system)  │
      └──────────────┬──────────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
    ┌───▼─────┐         ┌──────────▼─────────┐
    │ Agents  │         │   Tool Execution   │
    │         │         │                    │
    │ • Claude│         │ • File operations  │
    │ • Codex │         │ • Code analysis    │
    │ • Ollama│         │ • Git operations   │
    │         │         │ • Web search       │
    │         │         │ • MCP tools        │
    └─────────┘         └────────────────────┘
```

### Key Components

1. **LangGraph Orchestrator** (`codur/graph/main_graph.py`)
   - Three-phase planning: pattern-based → optional pre-classification → LLM planning
   - Execution routing: delegate to agents, run tools, or code modifications
   - Review loop: verification and automatic retry (max 10 iterations)

2. **Agents** (`codur/agents/`)
   - **Claude Code Agent** - Uses Claude Code CLI for complex tasks
   - **Codex Agent** - Uses OpenAI Codex with sandbox isolation
   - **Ollama Agent** - Local LLM execution via Ollama
   - **MCP Agents** - Specialized agents for MCP server tools

3. **Tool System** (`codur/tools/`)
   - 40+ integrated tools across 18 modules
   - File operations, code analysis, git, structured data, web search, MCP client

4. **Configuration** - YAML-based configuration for agents, LLMs, MCP servers

5. **CLI & TUI** - Command-line interface with basic Textual UI support

## Features

### Implemented ✅

**Orchestration & Planning**
- ✅ Three-phase planning system (pattern-based, optional pre-classification, LLM)
- ✅ Intelligent task routing to specialized agents
- ✅ Automatic retry loop with verification (up to 10 iterations)
- ✅ Multiple planning strategies (code fix, generation, refactoring, etc.)

**Agents**
- ✅ Claude Code agent (via CLI)
- ✅ Codex agent (with sandbox modes)
- ✅ Ollama agent (local LLM)
- ✅ MCP tool agents (Sheets, LinkedIn, custom)
- ✅ Sync and async execution for all agents

**Tools** (40+ functions)
- ✅ File operations (read, write, append, delete, move, copy)
- ✅ Code analysis (AST parsing, dependencies, quality metrics)
- ✅ Git operations (status, diff, log, commit, stage)
- ✅ Structured data (JSON, YAML, INI files)
- ✅ Web search and document fetching
- ✅ Python linting and validation

**Configuration & Extensibility**
- ✅ YAML-based configuration
- ✅ Multiple LLM providers (Anthropic, Groq, OpenAI, Ollama)
- ✅ Agent registry and dynamic registration
- ✅ Tool registry with extensibility

**Testing & Validation**
- ✅ Challenge-based test framework (9 challenges with expected outputs)
- ✅ Automated verification loop
- ✅ Unit tests for core components

**CLI Interface**
- ✅ `codur run` - Execute single task
- ✅ `codur configure` - Configure planning LLM and model listing
- ✅ Subcommands with verbose/raw output options

### Partially Implemented ⚠️

**Textual TUI**
- ⚠️ Basic command input and file search
- ⚠️ Split-pane layout structure
- ⚠️ Threading-based execution (not true async)
- ⚠️ Limited real-time progress display
- ⚠️ No user guidance injection during execution

**Async Support**
- ⚠️ Basic async/await with `aexecute()` methods
- ⚠️ ThreadPoolExecutor for timeouts (not native async)
- ⚠️ No concurrent agent execution

### Not Implemented ❌

- ❌ State checkpointing and resumption
- ❌ Pause/resume during execution
- ❌ User interjections mid-run (`:hint`, `:pause`, etc.)
- ❌ Concurrent multiple agent execution
- ❌ Full async TUI with concurrent input
- ❌ Persistent state across sessions

## Installation

```bash
# Install from source in development mode
pip install -e .

# Or with dev dependencies (includes testing tools)
pip install -e ".[dev]"

# Verify installation
codur --version
codur --help
```

### Requirements

- Python 3.10+
- ANTHROPIC_API_KEY (for Claude models via Claude Code or direct API)
- Optional: GROQ_API_KEY (for fast planning with Groq)
- Optional: OPENAI_API_KEY (for Codex agent)
- Optional: Ollama running locally (for local LLM execution)

## Configuration

Codur uses YAML configuration (default: `codur.yaml` in project root).

### Configuration Structure

The configuration includes:
- **mcp_servers** - MCP server definitions (Sheets, LinkedIn, custom)
- **agents** - Agent configurations and preferences
- **llm** - LLM provider profiles (Anthropic, Groq, OpenAI, Ollama)
- **runtime** - Orchestration settings (max iterations, timeouts, debug options)
- **tools** - Tool-specific settings (git write permissions, etc.)

### Quick Start Configuration

```yaml
# Minimal configuration - uses defaults
llm:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
    groq:
      api_key: ${GROQ_API_KEY}

runtime:
  max_iterations: 10
  verbose: false
```

### Example: Full Configuration

See `codur.yaml` in the repository for a complete configuration example with all options:
- LLM provider profiles (Groq, OpenAI, Anthropic, Ollama)
- Agent-specific configurations
- MCP server definitions
- Runtime behavior settings
- Tool enablement flags

### Environment Variables

Codur supports environment variable substitution in YAML:
```yaml
api_key: ${MY_API_KEY}  # Expands to environment variable value
```

Required environment variables:
- `ANTHROPIC_API_KEY` - For Claude Code agent and direct Claude API calls
- `GROQ_API_KEY` - For fast planning with Groq (recommended)
- `OPENAI_API_KEY` - For Codex agent (optional)
- Additional keys for MCP servers as needed

## Usage

### Quick Start

```bash
# Run a coding task
codur run "Write a Python function to calculate fibonacci numbers"

# With verbose output to see planning and execution details
codur -c "Fix the bug in auth.py" --verbose

# With custom configuration file
codur -c "Refactor the API module" --config ./my-config.yaml

# Raw output (minimal formatting)
codur -c "Generate unit tests" --raw
```

### Available Commands

#### `codur run <task>`
Execute a single task through the orchestrator.
```bash
codur run "Implement a REST API endpoint using FastAPI"
```

#### `codur -c/--command <task>`
Alternative syntax for running a task (no subcommand needed).
```bash
codur -c "Write a function that sorts a list"
codur --command "Debug the authentication module"
```

#### `codur configure`
Configure the planning LLM and explore available models.
```bash
# Set the default planning LLM
codur configure --llm-profile groq-qwen3-32b

# List available models from each provider
codur configure --list-models
codur configure --list-model-registry
```

### Global Options

```bash
# Verbose output (shows planning, execution details)
codur -c "task" --verbose
codur -c "task" -v

# Raw output (minimal formatting, no decorations)
codur -c "task" --raw

# Custom configuration file
codur -c "task" --config ./custom.yaml

# Limit LLM calls (for testing)
codur -c "task" --max-llm-calls 5

# Show help
codur --help
codur run --help

# Show version
codur --version
```

### TUI Mode (Experimental) ⚠️

```bash
# Launch the terminal UI (basic, threading-based)
codur tui
```

**Note:** The TUI is currently experimental with limited functionality:
- Basic command input and file search
- Threading-based execution (not true async)
- No real-time progress updates
- No pause/resume or user guidance injection

See the main code in `codur/tui.py` for the current TUI implementation.

## Development

### Project Structure

```
codur/
├── codur/                          # Main package
│   ├── cli.py                      # CLI entry point (Typer)
│   ├── config.py                   # Configuration management (Pydantic)
│   ├── llm.py                      # LLM factory and creation
│   ├── model_registry.py           # Model listing and API access
│   │
│   ├── agents/                     # Agent implementations (8 files)
│   │   ├── base.py                 # BaseAgent abstract class
│   │   ├── cli_agent_base.py       # Shared CLI agent logic
│   │   ├── claude_code_agent.py    # Claude Code via CLI
│   │   ├── codex_agent.py          # OpenAI Codex
│   │   └── ollama_agent.py         # Local Ollama LLM
│   │
│   ├── graph/                      # LangGraph orchestration
│   │   ├── main_graph.py           # Graph definition & execution
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── state_operations.py     # State manipulation helpers
│   │   ├── AGENTIC_LOGIC.md        # Detailed logic documentation
│   │   └── nodes/                  # 12 node implementations
│   │       ├── planning/           # Planning strategies (11 strategies)
│   │       ├── execution.py        # Agent execution & review
│   │       ├── tool_detection.py   # Smart tool pattern matching
│   │       └── ...                 # Other node implementations
│   │
│   ├── tools/                      # 40+ integrated tools (18 modules)
│   │   ├── filesystem.py           # File operations
│   │   ├── git.py                  # Git operations
│   │   ├── code_modification.py    # Code editing helpers
│   │   ├── python_ast.py           # AST analysis
│   │   ├── structured_data.py      # JSON/YAML/INI
│   │   ├── web.py                  # Web search & fetch
│   │   ├── mcp_tools.py            # MCP client
│   │   └── ...                     # Other tool modules
│   │
│   ├── providers/                  # LLM providers (5 modules)
│   │   ├── anthropic.py            # Anthropic Claude
│   │   ├── groq.py                 # Groq fast inference
│   │   ├── openai.py               # OpenAI GPT
│   │   └── ollama.py               # Ollama local
│   │
│   ├── tui.py                      # Textual TUI (experimental)
│   ├── tui_components.py           # TUI widgets
│   ├── tui_style.py                # TUI CSS styling
│   ├── constants.py                # Project constants
│   ├── observability/              # Metrics (minimal)
│   └── utils/                      # Utility modules
│
├── tests/                          # Test suite
│   ├── with_several_llm_calls/     # Challenge tests (9 challenges)
│   │   └── test_challenges.py      # Challenge runner
│   └── py_only/                    # Unit tests (50+ tests)
│       ├── test_ast_utils.py
│       ├── test_git.py
│       ├── test_mcp_tools.py
│       └── ...
│
├── challenges/                     # Challenge test cases
│   ├── 01-*/                       # Each with: prompt.txt, main.py, expected.txt
│   ├── 02-*/
│   └── ...
│
├── codur.yaml                      # Default configuration
├── pyproject.toml                  # Package metadata & dependencies
├── CLAUDE.md                       # Orchestrator guide & implementation details
├── README.md                       # This file
└── refactor_plan.md                # Future refactoring roadmap
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/py_only/test_git.py

# Run challenges
pytest tests/with_several_llm_calls/test_challenges.py

# Verbose output
pytest -v

# Show print statements
pytest -s
```

### Code Quality

```bash
# Format with black
black codur/ tests/

# Lint with ruff
ruff check codur/ tests/
ruff check --fix codur/ tests/  # Auto-fix issues
```

### Key Files to Understand

1. **`codur/graph/main_graph.py`** - Core orchestration logic (350-line graph definition)
2. **`codur/graph/AGENTIC_LOGIC.md`** - Detailed documentation of orchestration behavior
3. **`codur/graph/nodes/planning/strategies/`** - Task-specific planning strategies
4. **`codur/graph/nodes/tool_detection.py`** - Pattern-based tool detection (400+ lines)
5. **`codur/tools/filesystem.py`** - File operation tools and patterns
6. **`codur/agents/`** - Agent implementations and base classes

## How It Works

### Three-Phase Planning System

The orchestrator uses a multi-phase planning approach for intelligence and efficiency:

1. **Phase 0: Pattern Matching** (No LLM)
   - Fast classification using regex and heuristics
   - Detects simple patterns (file operations, greetings, etc.)
   - Zero API calls for common tasks

2. **Phase 1: Pre-Classification** (Optional, Fast LLM)
   - Quick classification with a small, fast LLM
   - Experimental feature, gated by configuration
   - Helps route to appropriate planning strategy

3. **Phase 2: Full Planning** (Groq or Anthropic)
   - Comprehensive task analysis
   - Strategy selection and prompt building
   - Agent routing decision

### Execution & Review Loop

After planning, the orchestrator either:
- **Delegates to an agent** (Claude Code, Codex, or Ollama)
- **Executes tools directly** (file operations, web search, etc.)
- **Modifies code** (with syntax validation)
- **Provides explanations** (for documentation requests)

The **review node** then:
- Verifies the result quality
- Compares output to expected format (when available)
- Automatically retries with improvements if needed (up to 10 iterations)

### Planning Strategies

Codur includes specialized strategies for:
- **Code fixes** - Debugging and bug fixes
- **Code generation** - New code from requirements
- **Refactoring** - Code improvement and reorganization
- **Explanation** - Documentation and understanding
- **File operations** - Create/move/delete files
- **Web search** - Research and information gathering
- And more...

## Architecture Decisions

### Why Three-Phase Planning?

1. **Efficiency** - Common tasks bypass expensive LLM calls
2. **Cost Control** - Groq for fast inference when needed
3. **Flexibility** - Different strategies for different task types
4. **Debugging** - Pattern phase easy to test and fix

### Agent Selection

- **Claude Code** (default) - Complex tasks requiring deep reasoning
- **Codex** - Code-specific work with sandbox safety
- **Ollama** - Local execution, privacy-sensitive tasks
- **MCP Tools** - Specialized operations (spreadsheets, etc.)

### Tool Integration

Tools are discovered and suggested automatically through:
- Pattern-based detection (detects file operations from text)
- Strategy-specific suggestions (each strategy knows relevant tools)
- JSON tool descriptions (LLM can select from available tools)

## Roadmap

### Current Status
- ✅ Core orchestration complete
- ✅ Multiple agents implemented
- ✅ 40+ tools integrated
- ✅ Challenge testing framework
- ⚠️ TUI experimental/limited

### Next Priorities
- 🔄 TUI improvements (true async, real-time updates)
- 🔄 State persistence (checkpointing, resumption)
- 🔄 Concurrent agents (multi-task execution)
- 🔄 User interjections (guidance mid-run)

## Design Principles

1. **Intelligence First** - Multi-phase planning balances speed vs quality
2. **Cost Conscious** - Pattern matching and Groq for efficiency before expensive APIs
3. **Single-Task Focus** - Complete understanding of one task before execution
4. **Automatic Improvement** - Retry loop with structured error feedback
5. **Extensible Architecture** - Easy to add agents, tools, and planning strategies
6. **Transparent Execution** - Verbose mode shows planning and reasoning

## Known Limitations

- **Single-task execution** - No parallel task handling
- **No state persistence** - Sessions don't checkpoint/resume
- **TUI experimental** - Limited to basic commands, not for production
- **Hardcoded timeouts** - CLI agents default to 10-second timeout
- **No user interjection** - Can't pause/resume or inject guidance during execution
- **Sequential agents** - Only one agent runs at a time

See `CLAUDE.md` for comprehensive documentation of limitations and design details.

## Contributing

Codur is experimental and under active development. Contributions welcome!

## Resources

- **[CLAUDE.md](./CLAUDE.md)** - Detailed orchestrator guide and limitations
- **[codur/graph/AGENTIC_LOGIC.md](./codur/graph/AGENTIC_LOGIC.md)** - Deep dive into planning and execution logic
- **[codur/tools/README.md](./codur/tools/README.md)** - Tool module documentation
- **[codur/graph/nodes/planning/strategies/README.md](./codur/graph/nodes/planning/strategies/README.md)** - How to create custom planning strategies
- **[refactor_plan.md](./refactor_plan.md)** - Planned improvements and refactoring

## Technology Stack

- **Orchestration**: LangGraph with Pydantic
- **CLI**: Typer with Rich for formatting
- **TUI**: Textual (experimental)
- **LLM Providers**: Anthropic, Groq, OpenAI, Ollama
- **Code Analysis**: AST, git, various linters
- **Configuration**: YAML with environment variable support
- **Testing**: Pytest with challenge framework

---

**Last Updated:** 2025-12-25
**Status:** Experimental - Not production ready
**Python**: 3.10+
