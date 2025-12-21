# Codur Agent Ecosystem

Codur orchestrates multiple specialized agents to handle different types of coding tasks efficiently and cost-effectively.

## Available Agents

### 1. Ollama (FREE Local LLM) 🦙

**Type:** MCP Server
**Cost:** FREE
**Best For:** Simple code generation

**Capabilities:**
- Generate boilerplate code
- Write simple functions
- Explain code concepts
- Quick prototypes

**When to Use:**
- "Write a hello world function"
- "Create a sorting algorithm"
- "Generate a REST API endpoint structure"

**Configuration:**
```yaml
ollama:
  type: "mcp"
  config:
    mcp_server: "ollama"
    model: "ministral-3:14b"
```

---

### 2. Claude Code 🤖

**Type:** Tool (CLI)
**Cost:** Claude tokens
**Best For:** Multi-file changes, complex reasoning

**Capabilities:**
- Multi-file refactoring
- Complex code analysis
- File system operations
- Tool usage (bash, read, write, grep)
- Architectural decisions
- Debugging complex issues

**When to Use:**
- "Refactor this authentication system across multiple files"
- "Add error handling to all API endpoints"
- "Analyze and fix the bug in the payment flow"
- "Redesign the database schema"

**Configuration:**
```yaml
claude_code:
  type: "tool"
  config:
    command: "claude"
    model: "sonnet"  # or "opus" for complex tasks
    max_tokens: 8000
```

**Special Features:**
- Can use tools (Bash, Read, Write, Edit)
- Context-aware across multiple files
- Can reason about architecture
- Recursive capability (Codur running in Claude Code can call Claude Code)

---

### 3. Codex 💻

**Type:** Tool (CLI)
**Cost:** OpenAI tokens (separate from Claude)
**Best For:** Code-specialized refactoring

**Capabilities:**
- Code refactoring
- Bug fixing
- Code optimization
- Test generation

**When to Use:**
- "Refactor this function for better performance"
- "Fix the memory leak in this class"
- "Add unit tests for this module"

**Configuration:**
```yaml
codex:
  type: "tool"
  config:
    command: "codex"
    model: "gpt-5-codex"
    reasoning_effort: "medium"
```

---

### 4. Google Sheets 📊

**Type:** MCP Server
**Cost:** Minimal API usage
**Best For:** Spreadsheet operations

**Capabilities:**
- Read/write Google Sheets data
- Update cells and ranges
- Batch operations

**When to Use:**
- "Read data from this spreadsheet"
- "Update Q4 sales figures"
- "Export this data to Google Sheets"

---

### 5. LinkedIn 💼

**Type:** MCP Server
**Cost:** Minimal API usage
**Best For:** Job searching

**Capabilities:**
- Search LinkedIn jobs
- Filter by location, type, remote
- Scrape job descriptions

**When to Use:**
- "Find Python jobs in the EU"
- "Search for remote DevOps positions"

---

## Routing Strategy

Codur intelligently routes tasks to the most appropriate agent:

### Decision Tree

```
Task Type               →  Agent        →  Cost
─────────────────────────────────────────────────
Simple code gen         →  Ollama       →  FREE
Multi-file changes      →  Claude Code  →  $$
Complex refactoring     →  Codex        →  $
Architectural design    →  Claude Code  →  $$
Spreadsheet ops         →  Sheets       →  $
Job search             →  LinkedIn     →  $
```

### Routing Rules (from `codur.yaml`)

```yaml
routing:
  simple: "ollama"              # FREE
  complex: "codex"               # Cheap
  multifile: "claude_code"       # Powerful
  reasoning: "claude_code"       # Best reasoning
  sheets: "sheets"               # Specialized
  jobs: "linkedin"               # Specialized
```

### Fallback Order

If the primary agent fails:
1. Ollama (try free first)
2. Claude Code (fallback to powerful)
3. Codex (last resort)

---

## Cost Optimization

### FREE Tasks (Use Ollama)
- Generate a function
- Write boilerplate
- Simple explanations
- Quick prototypes

### Low Cost (Use Codex)
- Single-file refactoring
- Bug fixes
- Optimization
- Test generation

### Medium Cost (Use Claude Code)
- Multi-file changes
- Architecture decisions
- Complex debugging
- Tool usage needed

---

## Agent Comparison

| Feature | Ollama | Claude Code | Codex |
|---------|--------|-------------|-------|
| **Cost** | FREE | $$ | $ |
| **Speed** | Medium | Medium | Fast |
| **Multi-file** | ❌ | ✅ | ❌ |
| **Tools** | ❌ | ✅ | ❌ |
| **Reasoning** | Good | Excellent | Good |
| **Code Quality** | Good | Excellent | Excellent |
| **Local** | ✅ | ❌ | ❌ |

---

## Recursive Architecture

Codur can call Claude Code, which can call Codur, creating powerful recursion:

```
┌────────────────────┐
│   User Request     │
└─────────┬──────────┘
          │
     ┌────▼────┐
     │  Codur  │ ◄────────┐
     └────┬────┘          │
          │               │
    ┌─────┴──────────┐    │
    │                │    │
┌───▼────┐    ┌─────▼────▼┐
│ Ollama │    │ Claude Code│
└────────┘    └─────┬──────┘
                    │
              ┌─────▼──────┐
              │   Tools    │
              │ Bash, Read │
              │ Write, etc.│
              └────────────┘
```

---

## Example Workflows

### Workflow 1: Simple Function (FREE)
```
User: "Write a fibonacci function"
Codur: Routes to Ollama
Ollama: Generates function
Result: FREE, fast, good quality
```

### Workflow 2: Multi-File Refactoring ($$)
```
User: "Refactor authentication across all controllers"
Codur: Routes to Claude Code
Claude Code: Analyzes multiple files, makes changes
Result: Comprehensive, high quality
```

### Workflow 3: Complex Project Setup ($$)
```
User: "Set up a new FastAPI project with auth"
Codur: Routes to Claude Code
Claude Code:
  1. Creates directory structure
  2. Writes multiple files
  3. Sets up dependencies
  4. Adds tests
  5. Creates documentation
Result: Complete, production-ready
```

### Workflow 4: Optimization ($ Codex)
```
User: "Optimize this database query"
Codur: Routes to Codex
Codex: Analyzes and optimizes
Result: Fast, efficient
```

---

## Adding New Agents

To add a new agent to Codur:

1. **Create agent class** in `codur/agents/`
2. **Update `codur.yaml`** with agent config
3. **Update routing** in `codur/graph/nodes.py`
4. **Add to CLI** in `codur/cli.py`

Example:

```python
# codur/agents/my_agent.py
class MyAgent:
    def __init__(self, config):
        self.config = config

    def execute(self, task: str) -> str:
        # Implementation
        pass

    async def aexecute(self, task: str) -> str:
        # Async implementation
        pass
```

---

## Best Practices

1. **Start Free**: Default to Ollama for simple tasks
2. **Be Specific**: Tell Codur which agent to use if you know
3. **Use Fallbacks**: Let Codur try multiple agents
4. **Monitor Costs**: Track which agents are used
5. **Combine Agents**: Use Ollama for drafts, Claude Code for refinement

## Commands

```bash
# Let Codur decide
codur run "Write a sorting function"

# Force specific agent
codur run "Use Ollama to write a sorting function"
codur run "Use Claude Code to refactor the auth system"

# List available agents
codur list-agents

# Interactive mode
codur tui  # TUI shows which agent is being used
```

---

**The ecosystem is designed for maximum efficiency and minimum cost while maintaining high quality!** 🚀
