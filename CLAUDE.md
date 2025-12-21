# Claude Code Integration 🤖

Comprehensive guide to using Claude Code agent in the Codur orchestrator for complex coding tasks, multi-file refactoring, and advanced reasoning.

---

## Table of Contents

- [Overview](#overview)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Agent Comparison](#agent-comparison)
- [Use Cases & Workflows](#use-cases--workflows)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

---

## Overview

### What is Claude Code Agent?

The **Claude Code Agent** is Codur's wrapper around the [Claude Code CLI](https://www.anthropic.com/claude/code) that enables intelligent routing of complex coding tasks to Claude's most capable models. It provides:

- ✅ **Multi-file editing** - Refactor across entire codebases
- ✅ **Tool usage** - Read, write, bash, grep, and more
- ✅ **Complex reasoning** - Architectural decisions and debugging
- ✅ **Async execution** - Non-blocking operations for TUI/interactive modes
- ✅ **Model selection** - Choose between Sonnet, Opus, or Haiku based on task complexity

### When to Use Claude Code vs Other Agents

| Scenario | Recommended Agent | Why |
|----------|------------------|-----|
| Simple code generation | Ollama | FREE, fast, good enough |
| Multi-file refactoring | **Claude Code** | Best multi-file support |
| Complex architecture | **Claude Code** | Superior reasoning |
| Code optimization | Codex | Specialized for optimization |
| Tool-assisted development | **Claude Code** | Full tool access |
| Offline development | Ollama | Runs locally |

### Position in the Agent Hierarchy

```
User Request → Plan Node (Orchestrator)
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    Ollama    Claude Code    Codex
    (FREE)      ($$)          ($)
```

The orchestrator automatically routes tasks to Claude Code when it detects:
- Multi-file changes needed
- Complex reasoning required
- File system operations
- Tool usage needed

---

## Setup & Installation

### Prerequisites

1. **Claude Code CLI** must be installed and available in your PATH
2. **Anthropic API key** configured

### Installation Steps

```bash
# 1. Install Claude Code CLI (if not already installed)
# Visit: https://www.anthropic.com/claude/code
# Follow installation instructions for your platform

# 2. Verify installation
claude --version

# 3. Configure API key
export ANTHROPIC_API_KEY="your-api-key-here"

# 4. Add to your shell profile for persistence (~/.bashrc, ~/.zshrc, etc.)
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc

# 5. Verify Codur can find Claude
codur list-agents
# Should show claude_code in the list
```

### Quick Test

```bash
# Test that Claude Code works
codur -c "Use Claude Code to create a hello world function in Python"
```

---

## Configuration

### Basic Configuration

Add to your `codur.yaml`:

```yaml
agents:
  profiles:
    claude_code:
      name: "claude_code"
      type: "tool"
      enabled: true
      config:
        command: "claude"        # Path to claude CLI
        model: "sonnet"          # Options: sonnet, opus, haiku
        max_tokens: 8000        # Maximum response length
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `command` | string | `"claude"` | Path to claude CLI binary (use full path if not in PATH) |
| `model` | string | `"sonnet"` | Which Claude model to use (see below) |
| `max_tokens` | int | `8000` | Maximum response length in tokens |

### Available Models

#### 🚀 Sonnet (Recommended)
```yaml
config:
  model: "sonnet"  # claude-sonnet-4.5
  max_tokens: 8000
```
- **Best for:** Most coding tasks
- **Speed:** Medium (5-15 seconds)
- **Cost:** ~$3 per million input tokens
- **Capabilities:** Excellent balance of speed and intelligence

#### 💎 Opus (Most Capable)
```yaml
config:
  model: "opus"  # claude-opus-4.5
  max_tokens: 16000
```
- **Best for:** Complex multi-file refactoring, architectural decisions
- **Speed:** Slower (10-30 seconds)
- **Cost:** ~$15 per million input tokens
- **Capabilities:** Maximum reasoning power

#### ⚡ Haiku (Fastest)
```yaml
config:
  model: "haiku"  # claude-haiku-4
  max_tokens: 4000
```
- **Best for:** Simple tasks, quick responses
- **Speed:** Fast (1-5 seconds)
- **Cost:** ~$0.25 per million input tokens
- **Capabilities:** Good for straightforward tasks

### Custom Agent Profiles

Create specialized profiles for different use cases:

```yaml
agents:
  profiles:
    # Heavy-duty refactoring agent
    claude-opus-refactor:
      name: "claude_code"
      config:
        model: "opus"
        max_tokens: 16000

    # Quick tasks agent
    claude-haiku-quick:
      name: "claude_code"
      config:
        model: "haiku"
        max_tokens: 4000

    # Custom path agent (if claude not in PATH)
    claude-custom:
      name: "claude_code"
      config:
        command: "/usr/local/bin/claude"
        model: "sonnet"
```

**Using custom profiles:**
```bash
codur -c "Use agent:claude-opus-refactor to refactor the entire auth system"
```

---

## Usage Examples

### Example 1: Simple Delegation

Let the orchestrator decide when to use Claude Code:

```bash
codur -c "Refactor the authentication system"
# Orchestrator detects multi-file task → routes to Claude Code
```

### Example 2: Explicit Agent Selection

Force Claude Code for a specific task:

```bash
codur run "Use Claude Code to analyze this codebase and suggest improvements"
```

### Example 3: Multi-File Refactoring

```bash
codur -c "Refactor all API endpoints to use async/await pattern"
# Claude Code will:
# 1. Read all relevant files
# 2. Make coordinated changes
# 3. Ensure consistency across files
```

### Example 4: Interactive Mode

```bash
codur interactive
```
```
codur> Use Claude Code to add error handling to all API endpoints
🤖 Analyzing codebase...
📝 Found 12 API endpoints
🔧 Adding try-catch blocks...
✅ Complete! All endpoints now have error handling.
```

### Example 5: TUI Mode (Real-time Interaction)

```bash
codur tui
```
Then type your request in the UI. Claude Code will execute asynchronously, allowing you to continue working.

### Example 6: Complex Architecture Task

```bash
codur -c "Design and implement a plugin system for the application"
# Claude Code will:
# 1. Analyze existing architecture
# 2. Design plugin interface
# 3. Create plugin loader
# 4. Add example plugins
# 5. Update documentation
```

---

## API Reference

### ClaudeCodeAgent Class

Located in: `/Users/sjuul/workspace/codur/codur/agents/claude_code_agent.py`

#### `__init__(config, override_config=None)`

Initialize the Claude Code agent.

**Parameters:**
- `config` (CodurConfig): Main configuration object
- `override_config` (dict, optional): Override specific settings

**Example:**
```python
from codur.config import load_config
from codur.agents.claude_code_agent import ClaudeCodeAgent

config = load_config()
agent = ClaudeCodeAgent(config)
```

#### `execute(task, context=None, timeout=600)`

Synchronously execute a task with Claude Code.

**Parameters:**
- `task` (str): The coding task to perform
- `context` (str, optional): Additional context to prepend to the task
- `timeout` (int, optional): Maximum execution time in seconds (default: 600)

**Returns:**
- `str`: Generated output from Claude

**Raises:**
- `FileNotFoundError`: If claude CLI is not installed
- `subprocess.TimeoutExpired`: If task exceeds timeout
- `Exception`: For other execution errors

**Example:**
```python
result = agent.execute(
    task="Refactor this function to be more efficient",
    context="Current implementation is O(n²)",
    timeout=300
)
print(result)
```

#### `aexecute(task, context=None, timeout=600)`

Asynchronously execute a task with Claude Code.

**Parameters:** Same as `execute()`

**Returns:**
- `str`: Generated output from Claude (as coroutine)

**Example:**
```python
import asyncio

async def main():
    result = await agent.aexecute("Create a REST API with FastAPI")
    print(result)

asyncio.run(main())
```

#### `execute_with_files(task, files, context=None)`

Execute a task with specific files in context.

**Parameters:**
- `task` (str): The coding task
- `files` (list[str]): List of file paths to include
- `context` (str, optional): Additional context

**Returns:**
- `str`: Generated output

**Example:**
```python
result = agent.execute_with_files(
    task="Refactor these authentication files",
    files=["auth/login.py", "auth/register.py", "auth/utils.py"]
)
```

#### `chat(messages, stream=False)`

Multi-turn conversation with Claude.

**Parameters:**
- `messages` (list[dict]): List of message dicts with `role` and `content`
- `stream` (bool, optional): Whether to stream response (default: False)

**Returns:**
- `str`: Assistant's response

**Example:**
```python
messages = [
    {"role": "user", "content": "What's wrong with this code?"},
    {"role": "assistant", "content": "I see a memory leak..."},
    {"role": "user", "content": "How do I fix it?"}
]
response = agent.chat(messages)
```

#### `_build_prompt(task, context)`

Internal helper to build the full prompt.

**Parameters:**
- `task` (str): The main task
- `context` (str): Additional context

**Returns:**
- `str`: Complete prompt with context prepended

---

## How It Works

### Execution Flow

```
User Request: "Refactor authentication system"
    ↓
Plan Node (LLM analyzes task)
    ├─ Detects: Multi-file changes needed
    ├─ Detects: Complex reasoning required
    └─ Decision: Route to Claude Code
    ↓
Delegate Node
    └─ Selected Agent: "claude_code"
    ↓
Execute Node
    ├─ Creates: ClaudeCodeAgent instance
    ├─ Resolves: Model config (sonnet/opus/haiku)
    └─ Calls: agent.execute(task)
    ↓
ClaudeCodeAgent.execute()
    ├─ Builds prompt with context
    ├─ Runs CLI: claude chat --model sonnet --max-tokens 8000 --message "<prompt>"
    ├─ Captures: stdout (result) and stderr (errors)
    └─ Returns: Generated code/response
    ↓
Review Node
    ├─ Checks: Result quality (TODO: currently accepts all)
    └─ Returns: Final response to user
    ↓
User receives: Complete, working solution
```

### Integration Points

**1. Planning System Prompt**

The orchestrator's planning prompt includes guidance for routing to Claude Code:

```
For code generation tasks requiring multi-file changes:
→ Use action: "delegate" with agent: "claude_code"
```

**2. Agent Resolution**

Claude Code can be referenced in multiple ways:
- `"claude_code"` - Direct agent name
- `"agent:claude_code"` - Explicit agent reference
- `"agent:claude-opus-refactor"` - Custom profile name

**3. Tool Usage**

When Claude Code executes, it has access to:
- `read_file` - Read files from workspace
- `write_file` - Write/update files
- `bash` - Execute shell commands
- `grep` - Search code patterns
- And more...

---

## Agent Comparison

### Feature Matrix

| Feature | Ollama 🦙 | Claude Code 🤖 | Codex 🔧 |
|---------|-----------|----------------|----------|
| **Cost** | FREE | $$ | $ |
| **Multi-file** | ❌ | ✅ | ❌ |
| **Tool Support** | ❌ | ✅ | ❌ |
| **Reasoning** | Good | Excellent | Good |
| **Speed** | Fast | Medium | Fast |
| **Local Execution** | ✅ | ❌ | ❌ |
| **Max Context** | 8k tokens | 200k tokens | 128k tokens |
| **Streaming** | ✅ | ❌ | ✅ |
| **Best For** | Simple code gen | Complex refactoring | Code optimization |

### When to Use Each Agent

#### Use Ollama When:
- ✅ Cost is a concern (FREE)
- ✅ Working offline
- ✅ Simple code generation
- ✅ Quick prototypes
- ✅ Learning/experimentation

#### Use Claude Code When:
- ✅ Multi-file refactoring needed
- ✅ Complex architectural changes
- ✅ Tool-assisted development
- ✅ Advanced reasoning required
- ✅ File system operations
- ✅ Quality matters more than cost

#### Use Codex When:
- ✅ Code optimization tasks
- ✅ Bug fixes
- ✅ Simpler than Claude Code requirements
- ✅ Moderate complexity tasks

---

## Use Cases & Workflows

### Use Case 1: Multi-File Refactoring

**Scenario:** Refactor authentication across multiple controllers

```bash
codur -c "Refactor authentication to use JWT tokens across all controllers"
```

**What Claude Code does:**
1. Reads all controller files
2. Analyzes current auth implementation
3. Creates new JWT auth system
4. Updates all controllers to use JWT
5. Adds tests and documentation
6. Returns comprehensive summary

**Cost:** ~$0.10 - $0.50 depending on codebase size

---

### Use Case 2: Complex Project Setup

**Scenario:** Set up new FastAPI project with best practices

```bash
codur run "Create a production-ready FastAPI project with:
- User authentication (JWT)
- Database integration (PostgreSQL)
- Docker configuration
- pytest test suite
- API documentation
- Error handling middleware"
```

**What Claude Code does:**
1. Creates directory structure
2. Writes all necessary files
3. Adds configuration files
4. Creates tests
5. Generates documentation
6. Provides setup instructions

**Cost:** ~$0.20 - $1.00

---

### Use Case 3: Debugging Complex Issues

**Scenario:** Find and fix a memory leak

```bash
codur -c "Analyze the codebase and find the source of memory leaks in the worker processes"
```

**What Claude Code does:**
1. Analyzes all worker-related code
2. Identifies potential leak sources
3. Explains the issue
4. Provides fixes
5. Suggests prevention strategies

**Cost:** ~$0.05 - $0.30

---

### Use Case 4: Code Review & Improvements

**Scenario:** Get architectural feedback

```bash
codur interactive
> Use Claude Code to review the codebase and suggest architectural improvements
```

**What Claude Code does:**
1. Analyzes codebase structure
2. Identifies anti-patterns
3. Suggests improvements
4. Provides refactoring plan
5. Estimates implementation effort

---

## Advanced Features

### Async Execution for Responsiveness

The TUI and interactive modes use `aexecute()` for non-blocking operations:

```python
async def handle_user_request(request):
    # User can continue working while Claude processes
    result = await agent.aexecute(request)
    return result
```

### Custom Context Injection

Add domain-specific context to improve results:

```python
context = """
Project uses:
- Python 3.11
- FastAPI framework
- PostgreSQL database
- Pytest for testing
- Follow PEP 8 style guide
"""

result = agent.execute(
    task="Create a new API endpoint",
    context=context
)
```

### Timeout Configuration

Adjust timeouts based on task complexity:

```python
# Quick task
result = agent.execute("Add docstring", timeout=60)

# Complex refactoring
result = agent.execute("Refactor entire API", timeout=1800)  # 30 minutes
```

### Error Handling Patterns

```python
try:
    result = agent.execute(task)
except FileNotFoundError:
    print("Claude CLI not installed. Please install: https://claude.com/claude-code")
except subprocess.TimeoutExpired:
    print("Task took too long. Try breaking it into smaller pieces.")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. "claude command not found"

**Problem:** Claude CLI is not installed or not in PATH

**Solution:**
```bash
# Check if claude is installed
which claude

# If not found, install from: https://www.anthropic.com/claude/code

# If installed but not in PATH, use full path in config
command: "/usr/local/bin/claude"
```

---

#### 2. "API key not configured"

**Problem:** ANTHROPIC_API_KEY environment variable not set

**Solution:**
```bash
# Set for current session
export ANTHROPIC_API_KEY="your-key-here"

# Add to shell profile for persistence
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc

# Verify
echo $ANTHROPIC_API_KEY
```

---

#### 3. Timeout errors on complex tasks

**Problem:** Task exceeds default 600-second timeout

**Solution:**

**Option A:** Increase timeout in code
```python
result = agent.execute(task, timeout=1800)  # 30 minutes
```

**Option B:** Use Opus model for complex tasks
```yaml
config:
  model: "opus"  # More capable, may finish faster
```

**Option C:** Break task into smaller pieces
```bash
# Instead of:
codur -c "Refactor entire codebase"

# Do:
codur -c "Refactor authentication module"
codur -c "Refactor API endpoints"
codur -c "Refactor database layer"
```

---

#### 4. Token limit exceeded

**Problem:** Response is cut off due to max_tokens limit

**Solution:**
```yaml
config:
  max_tokens: 16000  # Increase from default 8000
```

Or request smaller outputs:
```bash
codur -c "Refactor auth system (provide summary, not full code)"
```

---

#### 5. Poor quality results

**Problem:** Results don't meet expectations

**Solutions:**

1. **Be more specific:**
```bash
# Vague:
codur -c "Make the code better"

# Specific:
codur -c "Refactor the login function to:
- Add input validation
- Use async/await
- Add error handling
- Follow PEP 8 style"
```

2. **Add context:**
```python
context = "This is a FastAPI app using PostgreSQL. Follow existing patterns in auth/login.py"
result = agent.execute(task, context=context)
```

3. **Use Opus for complex tasks:**
```yaml
config:
  model: "opus"  # Better reasoning
```

---

## Best Practices

### 1. Choose the Right Model

```
Haiku  → Simple, straightforward tasks
Sonnet → Most tasks (recommended default)
Opus   → Complex multi-file refactoring only
```

**Cost-effective strategy:**
- Start with Sonnet
- Upgrade to Opus only if Sonnet struggles
- Use Haiku for trivial tasks (docstrings, comments)

---

### 2. Provide Clear Task Descriptions

**❌ Bad:**
```bash
codur -c "Fix the code"
```

**✅ Good:**
```bash
codur -c "Refactor the authentication function to:
1. Use JWT tokens instead of sessions
2. Add input validation
3. Add error handling for invalid credentials
4. Write unit tests"
```

---

### 3. Use Context Effectively

```python
# Include project-specific context
context = """
Stack: Python 3.11, FastAPI, PostgreSQL
Coding style: PEP 8, max line length 100
Testing: pytest with 80% coverage requirement
Error handling: Custom exceptions in exceptions.py
"""

result = agent.execute(task, context=context)
```

---

### 4. Set Appropriate Timeouts

```python
# Quick tasks: 1-2 minutes
agent.execute("Add docstring", timeout=120)

# Medium tasks: 5-10 minutes
agent.execute("Refactor module", timeout=600)

# Complex tasks: 15-30 minutes
agent.execute("Redesign architecture", timeout=1800)
```

---

### 5. Break Down Large Tasks

Instead of:
```bash
codur -c "Refactor entire application"
```

Do:
```bash
codur -c "Refactor authentication module"
codur -c "Refactor API endpoints"
codur -c "Refactor data models"
codur -c "Update tests for all changes"
```

Benefits:
- Faster feedback
- Easier to verify each step
- Lower cost (can use Sonnet instead of Opus)
- Better error recovery

---

### 6. Monitor Costs

Track your usage:
```bash
# Estimate before running
echo "Task estimated tokens: 5000 input + 3000 output"
echo "Sonnet cost: ~$0.024"
echo "Opus cost: ~$0.12"

# Choose accordingly
```

For high-volume usage, consider:
- Using Ollama for simple tasks (FREE)
- Batch similar requests
- Use Haiku for quick iterations

---

### 7. Version Control Integration

Always commit before major refactoring:
```bash
# Save current state
git commit -am "Before Claude Code refactoring"

# Run Claude Code
codur -c "Refactor authentication system"

# Review changes
git diff

# If good, commit
git commit -am "Refactored auth system with Claude Code"

# If bad, revert
git reset --hard HEAD^
```

---

## Performance Considerations

### Token Usage & Costs

**Pricing (as of 2024):**

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Typical Task Cost |
|-------|---------------------|----------------------|-------------------|
| Haiku | ~$0.25 | ~$1.25 | $0.01 - $0.05 |
| Sonnet | ~$3.00 | ~$15.00 | $0.05 - $0.30 |
| Opus | ~$15.00 | ~$75.00 | $0.20 - $2.00 |

**Example costs:**
- Add docstring to function: $0.01 (Haiku)
- Refactor single file: $0.10 (Sonnet)
- Multi-file refactoring: $0.50 (Sonnet) or $2.00 (Opus)
- Full codebase analysis: $1-5 (Opus)

---

### Response Times

**Average response times:**

| Model | Simple Task | Medium Task | Complex Task |
|-------|------------|-------------|--------------|
| Haiku | 1-3s | 3-5s | 5-10s |
| Sonnet | 3-5s | 5-15s | 15-30s |
| Opus | 5-10s | 10-30s | 30-60s |

**Factors affecting speed:**
- Task complexity
- Number of files to analyze
- Amount of code to generate
- API server load

---

### Optimization Tips

1. **Use the smallest model that works:**
   ```
   Haiku  → Docstrings, comments, simple fixes
   Sonnet → Most refactoring tasks
   Opus   → Complex multi-file changes only
   ```

2. **Limit file context:**
   ```python
   # Instead of analyzing entire codebase
   agent.execute_with_files(
       task="Refactor auth",
       files=["auth/login.py", "auth/utils.py"]  # Just what's needed
   )
   ```

3. **Use async for multiple tasks:**
   ```python
   async def process_multiple():
       tasks = [
           agent.aexecute("Task 1"),
           agent.aexecute("Task 2"),
           agent.aexecute("Task 3")
       ]
       results = await asyncio.gather(*tasks)
   ```

4. **Cache common patterns:**
   ```python
   # For repetitive tasks, save responses
   # and reuse with minor modifications
   ```

---

## Related Documentation

- [README.md](README.md) - Main Codur documentation
- [AGENT_ECOSYSTEM.md](AGENT_ECOSYSTEM.md) - All available agents
- [TEXTUAL_GUIDE.md](TEXTUAL_GUIDE.md) - TUI interface guide
- [Claude Code Official Docs](https://www.anthropic.com/claude/code) - Claude Code CLI documentation
- [Anthropic API Docs](https://docs.anthropic.com) - API reference and pricing

---

## Support & Contributing

**Issues?** Open an issue on GitHub: [Codur Issues](https://github.com/your-org/codur/issues)

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md)

**Questions?** Join our Discord: [Codur Community](https://discord.gg/codur)

---

**Last Updated:** 2025-12-20
**Version:** Codur 0.1.0
**Claude Code Version:** Latest (supports Sonnet 4.5, Opus 4.5, Haiku 4)
