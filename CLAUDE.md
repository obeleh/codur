# Claude Code Integration & Codur Orchestrator

Guide to using Codur's orchestrator framework with Claude Code, Codex, and Ollama agents for intelligent task delegation and execution.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Available Agents](#available-agents)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Limitations & TODOs](#limitations--todos)
- [Architecture](#architecture)
- [Best Practices](#best-practices)

---

## Overview

**Codur** is an orchestrator framework that routes coding tasks to the most appropriate agent (Claude Code, Codex, or Ollama). It uses a LangGraph-based planning → execution → review loop to handle complex tasks with error recovery.

### Core Capabilities

- ✅ **Task routing** - LLM-based planning decides which agent to use
- ✅ **Multi-agent execution** - Claude Code, Codex, and Ollama agents
- ✅ **Error recovery** - Automatic retry loop (up to 10 iterations by default)
- ✅ **Async support** - Non-blocking execution for all agents
- ✅ **Configuration-driven** - Extensive YAML configuration for agents and LLMs
- ✅ **CLI interface** - Typer-based CLI for easy task submission

### Key Limitation

Codur is a **single-task orchestrator**. It executes one task per invocation and routes it to the best agent. It does NOT maintain stateful conversations or allow user interjection during execution.

---

## How It Works

### The Orchestration Flow

```
User Task
   ↓
[Plan Node] ← LLM analyzes task, decides on action
   ↓
   ├→ "delegate" → [Delegate Node] → [Execute Node] → [Review Node]
   ├→ "tool"     → [Tool Runner]    → [Review Node]
   └→ "respond"  → [Direct Response]
   ↓
[Review/Continue Loop]
   ├→ if verification fails & iterations < 10:  go back to Plan
   └→ else: return final response
   ↓
Result
```

### What Happens in Each Node

1. **Plan Node** - LLM with JSON mode reads the task and decides:
   - `action`: "delegate" (route to agent), "tool" (run tool), or "respond" (answer directly)
   - `agent`: which agent to use (if delegating)
   - `reasoning`: why this choice was made

2. **Delegate Node** - Routes to the selected agent based on config

3. **Execute Node** - Runs the agent (Claude Code, Codex, or Ollama)
   - Captures output and errors
   - Applies timeout (default 10 seconds for CLI tools)

4. **Review Node** - Checks if the result is acceptable
   - For fix/debug tasks: runs `python main.py` and compares to `expected.txt`
   - Decides to continue (retry) or end
   - Can attempt local repairs for common issues (off-by-one errors, etc.)

### Error Recovery Loop

If verification fails and iterations < 10:
1. Creates structured error message with:
   - Expected vs actual output
   - Current implementation (main.py)
   - Error details from stderr
2. Prunes message history to prevent context explosion
3. Sends back to planning node for retry

---

## Available Agents

### 1. Claude Code Agent

**What it does**: Wraps the Claude Code CLI for AI-powered code generation and refactoring.

**Location**: `codur/agents/claude_code_agent.py`

**Capabilities**:
- Multi-file editing (via Claude Code CLI)
- Complex reasoning and architecture decisions
- Full file system access through Claude Code tools

**Supported Methods**:
```python
agent.execute(task, timeout=600)              # Sync execution
agent.aexecute(task, timeout=600)             # Async execution
agent.execute_with_files(task, files=[...])   # With file context
agent.chat(messages)                          # Multi-turn (⚠️ TODO: history not maintained)
```

**Configuration** (codur.yaml):
```yaml
agents:
  claude_code:
    name: "claude_code"
    type: "tool"
    config:
      command: "claude"  # Path to claude CLI
      model: "sonnet"    # sonnet, opus, or haiku
```

**Limitations**:
- ⚠️ `chat()` doesn't maintain conversation history (only uses last user message)
- Requires Claude Code CLI installed and `ANTHROPIC_API_KEY` env var
- Timeout default is 10 seconds in CLI execution context

**Known Issues**:
- Large error messages can get truncated when passed through planning LLM
- Context window can fill up with error messages in long retry loops

---

### 2. Codex Agent

**What it does**: Wraps OpenAI Codex CLI for code analysis and optimization.

**Location**: `codur/agents/codex_agent.py`

**Capabilities**:
- Code analysis and suggestions
- Sandbox execution modes (read-only, workspace-write, danger-full-access)
- Resume last session via `resume_last()`

**Supported Methods**:
```python
agent.execute(task, timeout=600)      # Sync execution
agent.aexecute(task, timeout=600)     # Async execution
agent.resume_last()                   # Resume previous session
```

**Configuration** (codur.yaml):
```yaml
agents:
  codex:
    name: "codex"
    type: "tool"
    config:
      command: "codex"          # Path to codex CLI
      sandbox: "workspace-write"  # Sandbox mode
```

**Limitations**:
- ⚠️ Streaming (`astream()`) not implemented - raises `NotImplementedError`
- Requires `OPENAI_API_KEY` env var
- Sandbox modes need careful testing for your use case

---

### 3. Ollama Agent

**What it does**: Runs local LLMs via Ollama for offline code generation.

**Location**: `codur/agents/ollama_agent.py`

**Capabilities**:
- Completely local/offline execution (no API keys needed)
- Model switching via `switch_model()`
- Streaming support via `astream()`

**Supported Methods**:
```python
agent.execute(task, timeout=60)     # Sync execution
agent.aexecute(task, timeout=60)    # Async execution
agent.astream(task)                 # Streaming execution
agent.switch_model(model_name)      # Switch models
```

**Configuration** (codur.yaml):
```yaml
agents:
  ollama:
    name: "ollama"
    type: "local"
    config:
      base_url: "http://localhost:11434"
      default_model: "ministral-3:14b"
```

**Limitations**:
- Requires Ollama service running locally
- Model quality depends on chosen model
- No API-based reasoning capabilities

---

## Configuration

### Available LLM Providers for Orchestrator

The planning node can use different LLMs. Currently configured providers:

1. **Groq** - Fast inference (default orchestrator LLM)
   - Models: `qwen/qwen3-32b` (default), `llama-3.3-70b`
   - Requires: `GROQ_API_KEY`
   - JSON mode: ✅ Supported

2. **Anthropic** - Claude family
   - Requires: `ANTHROPIC_API_KEY`
   - JSON mode: ⚠️ Uses prompt-based instead of native

3. **OpenAI** - GPT models
   - Requires: `OPENAI_API_KEY`
   - JSON mode: ✅ Supported

4. **Ollama** - Local models
   - No API key needed
   - Default: `ministral-3:14b`

### Core Configuration (codur.yaml)

```yaml
# Orchestrator LLM (for planning decisions)
llm:
  default_profile: "groq-qwen3-32b"  # What LLM makes routing decisions
  providers:
    groq:
      api_key: ${GROQ_API_KEY}

# Agent routing preferences
agents:
  preferences:
    default_agent: "agent:groq-qwen3-32b"  # Fallback agent
    routing:
      "fix":     "agent:groq-qwen3-32b"
      "debug":   "agent:groq-qwen3-32b"
      "refactor": "agent:groq-qwen3-32b"

# Orchestrator behavior
runtime:
  max_iterations: 10          # Retry limit for fix tasks
  detect_tool_calls_from_text: true  # Parse tools from LLM text
```

### Model Names Currently in Config

⚠️ **Note**: Some model names in `codur.yaml` are placeholder/fictional and don't correspond to real models:
- `gpt-5` - Not a real OpenAI model
- `gpt-5-codex` - Not a real OpenAI model
- `gpt-5-mini` - Not a real OpenAI model

Use actual models like `gpt-4-turbo`, `gpt-3.5-turbo`, `claude-opus-4-1`, etc.

---

## Usage Examples

### Example 1: Run a Task with Default Agent

```bash
codur -c "Fix the bug in main.py where the title case function isn't working"
```

The orchestrator will:
1. Use planning LLM to decide on action
2. Delegate to appropriate agent
3. Run verification if available
4. Retry up to 10 times if verification fails

### Example 2: Use Claude Code Directly

```bash
codur -c "Use claude_code to refactor the authentication module for better security"
```

### Example 3: Use Ollama (Offline)

```bash
codur -c "Use ollama to generate a Python function that validates email addresses"
```

Requires Ollama running locally:
```bash
ollama run ministral-3:14b
```

### Example 4: Verbose Output

```bash
codur -c "Fix main.py" --verbose
```

Shows planning decisions, agent selection, and retry details.

---

## Limitations & TODOs

### Current Limitations

1. **Single-task execution only** - One task per invocation, no stateful conversations
2. **No user interaction** - Can't ask clarifying questions or pause for user input
3. **Message history truncation** - Long error messages get truncated when sent to planning LLM
4. **Hardcoded timeouts** - CLI agents have 10-second timeout, not configurable per agent

### Unimplemented Features (TODOs in code)

1. **ClaudeCodeAgent.chat()** - Multi-turn conversation with history
   - Currently only uses last user message
   - Full conversation history not maintained

2. **CodexAgent.astream()** - Streaming responses
   - Raises `NotImplementedError`
   - Sync and async execution work, but not streaming

3. **Review node quality verification** - Currently accepts all results
   - Only checks exit codes and output matching
   - No code quality or logic verification

4. **MCP Server integration** - LinkedIn and Sheets MCP servers configured but not integrated

5. **TUI/Interactive mode** - Marked as "async" but infrastructure incomplete

---

## Architecture

### Directory Structure

```
codur/
├── agents/                 # Agent implementations
│   ├── base.py            # BaseAgent and BaseCLIAgent classes
│   ├── claude_code_agent.py
│   ├── codex_agent.py
│   └── ollama_agent.py
├── providers/             # LLM provider integrations
│   ├── anthropic.py
│   ├── groq.py
│   ├── openai.py
│   └── ollama.py
├── graph/                 # LangGraph orchestration
│   ├── main_graph.py      # Main graph definition (recursion_limit=60)
│   ├── state.py           # AgentState definition
│   └── nodes/             # Individual graph nodes
│       ├── execution.py   # Execute and Review nodes
│       ├── planning/      # Planning node implementation
│       └── routing.py     # Conditional edges
├── cli.py                 # Typer CLI interface
├── config.py              # Pydantic configuration
└── codur.yaml            # Runtime configuration
```

### Key Classes

**BaseAgent** (`agents/base.py`)
```python
class BaseAgent:
    async def execute(task, timeout) → str
    async def aexecute(task, timeout) → str
```

**AgentState** (`graph/state.py`)
```python
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    agent_outcome: dict
    final_response: str
    iterations: int
    next_action: str  # "continue", "end", "delegate", "tool"
    config: CodurConfig
```

---

## Best Practices

### Coding Guidelines

- Use the `console` module for logging (from Rich).
- Avoid local imports; keep imports at the top of files.

### 1. Task Structure for Fix/Debug

For the best results with automatic verification and retry:

1. Include a `main.py` file with test cases
2. Create `expected.txt` with correct output
3. Use keywords in your task: "fix", "debug", "bug", "error"

Example structure:
```
.
├── prompt.txt      # Task description
├── main.py         # Implementation (gets modified)
└── expected.txt    # Expected output for verification
```

The orchestrator will automatically:
- Run `python main.py` after each agent attempt
- Compare output to `expected.txt`
- Retry if mismatch (up to 10 times)
- Show actual vs expected in error messages

### 2. Writing Good Prompts

**❌ Vague**:
```
Fix the code
```

**✅ Specific**:
```
Fix the title_case function in main.py to:
1. Capitalize the first letter of each word
2. Handle hyphenated words (state-of-the-art → State-of-the-Art)
3. Preserve proper nouns like NASA
```

### 3. Agent Selection Strategy

- **Claude Code**: Multi-file refactoring, complex architecture
- **Codex**: Code analysis, optimization suggestions
- **Ollama**: Simple tasks, when offline/cost is concern

Let the orchestrator decide automatically by omitting agent name. Or force it:
```bash
codur -c "Use claude_code to ..."
codur -c "Use codex to ..."
codur -c "Use ollama to ..."
```

### 4. Error Message Handling

The orchestrator now provides structured error messages including:
- Expected vs actual output (truncated to ~20 lines)
- Current implementation being tested
- Stderr/exception details
- Clear action instructions

This improves retry accuracy but may not cover all error scenarios.

---

## Troubleshooting

### Claude Code CLI not found

```bash
# Install Claude Code CLI
# Visit: https://www.anthropic.com/claude/code

# Verify installation
claude --version

# Check it's in PATH
which claude
```

### Groq API errors

```bash
# Set API key
export GROQ_API_KEY="your-key-here"

# Verify in shell
echo $GROQ_API_KEY
```

### Ollama not responding

```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull ministral-3:14b

# Test it
ollama run ministral-3:14b "hello"
```

### Verification stuck in retry loop

- Check that `expected.txt` exists and is correct
- Look at actual vs expected output in verbose mode
- May need to increase `max_iterations` in config if legitimate test case

---

## Related Documentation

- [README.md](README.md) - Main Codur overview
- [AGENT_ECOSYSTEM.md](AGENT_ECOSYSTEM.md) - Detailed agent specifications
- [Claude Code Official Docs](https://www.anthropic.com/claude/code) - Claude Code CLI guide
- [LangGraph Docs](https://python.langchain.com/docs/langgraph) - Graph orchestration framework

---

**Last Updated:** 2025-12-21
**Status:** Actively maintained
**Note:** This documentation reflects the actual Codur implementation. For planned features, see issues and TODOs in code.
