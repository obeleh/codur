# Codur - Autonomous Coding Agent Orchestrator

**Status:** 🚧 Experimental / In Development

An autonomous coding agent built with LangGraph that orchestrates tasks by delegating to specialized agents and MCP servers.

## Vision

Codur is your main coding loop - an intelligent orchestrator that:
- Analyzes coding tasks and chooses the best approach
- Delegates to cost-effective agents (Ollama for simple tasks, Codex for complex work)
- Integrates with MCP servers (Google Sheets, LinkedIn, custom tools)
- Allows real-time user guidance and interruptions during execution
- Manages state across concurrent operations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Codur CLI                           │
│               (Async User Interface)                    │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────▼──────────┐
        │   LangGraph Agent    │
        │   (Orchestrator)     │
        └───────┬──────────────┘
                │
        ┌───────┴────────────────────────┐
        │                                │
    ┌───▼────┐              ┌───────▼────────┐
    │ Agents │              │  MCP Servers   │
    │        │              │                │
    │ Ollama │              │  Ollama        │
    │ Codex  │              │  Sheets        │
    │        │              │  LinkedIn      │
    └────────┘              └────────────────┘
```

### Key Components

1. **LangGraph Orchestrator** - State machine that plans, delegates, executes, and reviews
2. **Agent Wrappers** - Interfaces to Ollama, Codex, and other coding agents
3. **MCP Clients** - Connect to MCP servers for specialized tools
4. **Async Runtime** - Concurrent execution with user interjections
5. **YAML Configuration** - Flexible configuration for agents, MCP servers, and runtime

## Features

### Current

✅ **Textual TUI** - Beautiful async terminal interface
✅ CLI interface with Typer and Rich
✅ YAML-based configuration system
✅ Agent routing (Ollama, Codex, MCP servers)
✅ LangGraph state machine
✅ Configuration management
✅ Complete agent implementations (sync + async)

### Planned (From Codex Analysis)

🚧 **Async Execution**
- Convert to `async/await` with `ainvoke` and `astream_events`
- Concurrent user input while agent runs
- Real-time progress streaming

🚧 **User Interjections**
- Pause/resume during execution
- Inject guidance mid-run (`:hint`, `:pause`, `:set agent=codex`)
- Interactive control without restarting

🚧 **State Management**
- Persistent checkpoints with LangGraph MemorySaver
- Concurrent operation tracking
- Resume from interruptions

🚧 **Enhanced UI**
- Live progress display with Rich
- Multiple panes: current step, active agents, logs
- Async input with prompt_toolkit

## Installation

```bash
cd /Users/sjuul/workspace/mcp-servers/codur

# Install in development mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Configuration

Codur uses YAML configuration. Create `codur.yaml` in your project root or `~/.codur/config.yaml`.

### Example Configuration

```yaml
# MCP Servers
mcp_servers:
  ollama:
    command: "/path/to/.venv/bin/python"
    args: ["/path/to/ollama/mcp_server.py"]
    cwd: "/path/to/ollama"
    env:
      OLLAMA_HOST: "http://localhost:11434"

# Agents
agents:
  enabled: ["ollama", "codex"]
  preferences:
    default_agent: "ollama"
    routing:
      simple: "ollama"      # Simple code → Free local LLM
      complex: "codex"       # Complex work → Codex
      sheets: "sheets"       # Spreadsheets → Sheets MCP
    fallback_order: ["ollama", "codex"]

# LLM for orchestration
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-5-20250929"
  temperature: 0.7
  api_keys:
    anthropic_env: "ANTHROPIC_API_KEY"

# Runtime settings
runtime:
  max_iterations: 10
  verbose: false
  async:
    max_concurrent_agents: 3
    stream_events: true
    user_input_poll_ms: 100
```

See `codur.yaml` for a complete example.

## Usage

### TUI Mode (Recommended) ⭐

```bash
# Launch the interactive TUI
codur tui

# Beautiful split-pane interface with:
# - Real-time agent progress (top pane)
# - Live command input (bottom pane)
# - Pause/resume agents mid-execution
# - Provide hints and guidance while running
```

See [TEXTUAL_GUIDE.md](./TEXTUAL_GUIDE.md) for full TUI documentation.

### Basic Command

```bash
# Run a single task
codur run "Write a Python function to calculate fibonacci numbers"

# With verbose output
codur run "Refactor this authentication module" --verbose

# With custom config
codur run "Create a REST API endpoint" --config ./my-config.yaml
```

### Interactive Mode (Simple)

```bash
# Start basic interactive session
codur interactive

codur> Write a function to sort a list
codur> Use Ollama to generate a hello world
codur> quit
```

### List Available Resources

```bash
# List configured agents
codur list-agents

# List MCP servers
codur list-mcp

# Show version
codur version
```

## Development

### Project Structure

```
codur/
├── codur/
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   ├── config.py           # Configuration management
│   ├── agents/             # Agent wrappers
│   │   ├── ollama_agent.py
│   │   └── codex_agent.py
│   ├── graph/              # LangGraph components
│   │   ├── main_graph.py   # Graph definition
│   │   ├── nodes.py        # Graph nodes
│   │   └── state.py        # State schema
│   ├── tools/              # Tool integrations
│   └── mcp_clients/        # MCP client wrappers
├── codur.yaml              # Example configuration
├── pyproject.toml          # Package definition
└── README.md               # This file
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format
black codur/

# Lint
ruff check codur/
```

## Async Architecture (Codex Analysis)

Based on Codex review, the async architecture will support:

### Concurrent Execution

```python
async def run_agent(prompt, cfg):
    graph = create_agent_graph(cfg)

    async with asyncio.TaskGroup() as tg:
        # Graph execution task
        tg.create_task(graph_worker())

        # User input task (can interject anytime)
        tg.create_task(input_worker())
```

### User Interjections

```bash
codur> Create a REST API
[Agent planning...]
:pause              # User interrupts
[Agent paused]
:hint use FastAPI instead of Flask
[Agent resumes with new guidance]
:set agent=codex    # Switch agents mid-execution
```

### State Management

- **Checkpointing**: LangGraph MemorySaver for persistence
- **Concurrent operations**: Track multiple agent executions
- **Resume capability**: Pick up from interruptions

## Roadmap

### Phase 1: Core Functionality (Current)
- [x] CLI interface
- [x] YAML configuration
- [x] Basic LangGraph orchestration
- [ ] Agent routing logic
- [ ] MCP client implementations

### Phase 2: Async & Interactivity
- [ ] Convert to async/await
- [ ] Streaming events
- [ ] User interjections
- [ ] Live progress UI

### Phase 3: Advanced Features
- [ ] Multi-agent concurrency
- [ ] Checkpointing & resume
- [ ] Tool composition
- [ ] Quality checks (tests, lint)

### Phase 4: Polish
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Error handling
- [ ] Performance optimization

## Design Principles

1. **Cost-Effective**: Route simple tasks to free local LLMs (Ollama)
2. **Autonomous**: Make smart decisions about which agent to use
3. **Interruptible**: Users can guide execution in real-time
4. **Extensible**: Easy to add new agents and MCP servers
5. **Transparent**: Show what's happening at each step

## Contributing

This is an experimental project. Contributions and ideas welcome!

## Credits

- **LangGraph**: Agent orchestration framework
- **Anthropic Claude**: Main orchestrator LLM
- **Ollama**: Local LLM execution
- **OpenAI Codex**: Code-specialized tasks

---

**Last Updated:** 2025-12-19
**Status:** Experimental - Not production ready
