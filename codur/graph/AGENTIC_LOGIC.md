# Agentic Logic Reference

This document describes the current implementation of Codur's agentic orchestration system. Use this as a reference when working on the graph, nodes, or agent execution logic.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Graph Flow](#graph-flow)
3. [State Management](#state-management)
4. [Node Implementations](#node-implementations)
5. [Agent Execution](#agent-execution)
6. [Tool Detection & Execution](#tool-detection--execution)
7. [Verification & Retry Loop](#verification--retry-loop)
8. [Configuration](#configuration)

---

## Architecture Overview

Codur uses **LangGraph** to orchestrate an agentic workflow. The system follows a plan-delegate-execute-review pattern with automatic retry loops for fix tasks.

```
┌─────────────────────────────────────────────────────────────────┐
│                        GRAPH FLOW                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌────────────┐     ┌─────────┐               │
│   │  PLAN    │────►│  DELEGATE  │────►│ EXECUTE │               │
│   └──────────┘     └────────────┘     └────┬────┘               │
│        │                                    │                    │
│        │                                    ▼                    │
│        │           ┌──────────┐        ┌────────┐               │
│        │◄──────────│ continue │◄───────│ REVIEW │               │
│        │           └──────────┘        └───┬────┘               │
│        │                                   │                     │
│        ▼                                   │ end                 │
│   ┌──────────┐                             ▼                    │
│   │   TOOL   │─────────────────────►  ┌────────┐                │
│   └──────────┘                        │  END   │                │
│                                       └────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `main_graph.py` | Graph definition, node wiring, entry point |
| `state.py` | AgentState TypedDict definition |
| `nodes/routing.py` | `should_delegate()` and `should_continue()` routing functions |
| `nodes/planning/core.py` | PlanningOrchestrator - analyzes task and decides action |
| `nodes/execution.py` | AgentExecutor, delegate_node, execute_node, review_node |
| `nodes/tools.py` | Tool execution node |
| `nodes/tool_detection.py` | Pattern-based tool call detection from text |
| `nodes/non_llm_tools.py` | Fast-path tool detection without LLM |

---

## Graph Flow

### Entry Point
The graph starts at the **plan** node. See `main_graph.py:55`.

### Nodes

1. **plan** → Analyzes task, decides action (delegate/tool/end)
2. **delegate** → Routes to selected agent
3. **execute** → Runs the agent with tool-using loop
4. **tool** → Executes filesystem/MCP tools directly
5. **review** → Verifies result, decides to loop or end

### Edges

```python
# From plan node (conditional)
plan → delegate    # if next_action == "delegate"
plan → tool        # if next_action == "tool"
plan → END         # if next_action == "end"

# Fixed edges
delegate → execute
execute → review
tool → review

# From review node (conditional)
review → plan      # if next_action == "continue" (retry loop)
review → END       # if next_action == "end"
```

### Recursion Limit

The graph is compiled with `recursion_limit=300` to support:
- 10 max review iterations
- 5 tool iterations per agent execution
- Safety margin for complex tasks

---

## State Management

### AgentState (TypedDict)

Defined in `state.py`:

```python
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]  # Conversation history (accumulates via operator.add)
    next_action: str                 # "delegate" | "tool" | "end" | "continue"
    agent_outcome: dict              # Result from last agent action
    iterations: int                  # Current iteration count
    final_response: str              # Final output to return
    selected_agent: str              # Agent selected for delegation
    tool_calls: list[dict]           # Tool calls requested by planner
    verbose: bool                    # Enable verbose logging
    config: CodurConfig              # Runtime configuration
```

### Message Accumulation

The `messages` field uses `Annotated[Sequence[BaseMessage], operator.add]` which means messages are **accumulated** across nodes, not replaced.

### Message Pruning

To prevent context explosion during retry loops, `_prune_messages()` in `execution.py:417` keeps:
- Original HumanMessage (the task)
- Last 4 verification error messages
- Discards older intermediate messages

---

## Node Implementations

### 1. Plan Node

**Location:** `nodes/planning/core.py` → `PlanningOrchestrator.plan()`

**Purpose:** Analyze the task and decide what action to take.

**Flow:**
1. Check for non-LLM tool detection (fast path) via `run_non_llm_tools()`
2. If no fast-path match, call LLM with planning prompt
3. Parse LLM response as JSON decision
4. Handle fallbacks if parsing fails

**Decision Format:**
```json
{
    "action": "delegate" | "tool" | "respond" | "done",
    "agent": "agent:ollama" | "llm:groq-qwen3-32b" | null,
    "reasoning": "brief explanation",
    "response": "only for greetings",
    "tool_calls": [{"tool": "read_file", "args": {"path": "..."}}]
}
```

**Returns:** `PlanNodeResult` with `next_action`, `selected_agent`, `tool_calls`, etc.

---

### 2. Delegate Node

**Location:** `nodes/execution.py:222` → `delegate_node()`

**Purpose:** Route task to the selected agent.

**Logic:**
```python
selected_agent = state.get("selected_agent", default_agent)
return {"agent_outcome": {"agent": selected_agent, "status": "delegated"}}
```

Simply passes the selected agent to the execute node.

---

### 3. Execute Node

**Location:** `nodes/execution.py:247` → `execute_node()` → `AgentExecutor`

**Purpose:** Actually run the delegated agent.

**Agent Resolution:**
1. `agent:name` → Registered agent class (ollama, codex, claude_code)
2. `llm:profile` → LLM profile from config
3. Falls back to `_execute_llm_agent()` for LLM-type agents

**Tool-Using Loop:**
The `_execute_agent_with_tools()` method wraps agent execution:

```python
while tool_iteration < max_tool_iterations:  # default 5
    result = agent.execute(task)
    tool_calls = _TOOL_DETECTOR.detect(result)
    if not tool_calls:
        return result  # Done
    tool_results = self._execute_tool_calls(tool_calls)
    # Feed results back to agent for next iteration
    messages.append(AIMessage(content=result))
    messages.append(SystemMessage(content=f"Tool results:\n{tool_results}"))
    tool_iteration += 1
```

---

### 4. Tool Node

**Location:** `nodes/tools.py:56` → `tool_node()`

**Purpose:** Execute filesystem and MCP tools directly (not via agent).

**Supported Tools:**
- File operations: `read_file`, `write_file`, `append_file`, `delete_file`, `copy_file`, `move_file`
- Search: `search_files`, `grep_files`, `list_files`, `list_dirs`, `file_tree`
- Structured data: `read_json`, `write_json`, `read_yaml`, `write_yaml`, `read_ini`, `write_ini`
- Git: `git_status`, `git_diff`, `git_log`, `git_stage_files`, `git_commit`
- MCP: `list_mcp_tools`, `call_mcp_tool`, `read_mcp_resource`
- Web: `fetch_webpage`, `duckduckgo_search`

---

### 5. Review Node

**Location:** `nodes/execution.py:260` → `review_node()`

**Purpose:** Verify result and decide whether to loop or end.

**Flow:**
1. Check if this is a "fix task" (keywords: fix, bug, implement, create, etc.)
2. If fix task → run `_verify_fix()`:
   - Execute `python main.py`
   - Compare output to `expected.txt`
3. If verification fails:
   - Try `_attempt_local_repair()` (pattern-based mutations)
   - If repair fails, build error message and set `next_action: "continue"`
4. If verification passes or not a fix task → set `next_action: "end"`

**Verification Error Message Structure:**
```
Verification failed: Output does not match expected.

=== Expected Output ===
<truncated expected>

=== Actual Output ===
<truncated actual>

=== Error/Exception ===
<stderr if any>

=== Current Implementation (main.py) ===
```python
<code if small enough>
```

=== Action ===
Analyze the output mismatch and fix the implementation.
```

---

## Agent Execution

### Agent Types

| Type | Resolution | Example |
|------|------------|---------|
| `agent:name` | AgentRegistry lookup | `agent:ollama`, `agent:claude_code` |
| `llm:profile` | LLM profile from config | `llm:groq-qwen3-32b` |
| Config type=llm | Match model to profile | `groq-qwen3-32b` in agents.configs |
| Config type=tool | Registered agent class | `claude_code` |

### Registered Agents

Defined in `codur/agents/`:

1. **OllamaAgent** (`ollama_agent.py`) - Local Ollama models
2. **CodexAgent** (`codex_agent.py`) - OpenAI Codex CLI
3. **ClaudeCodeAgent** (`claude_code_agent.py`) - Claude Code CLI

Agents are registered via `AgentRegistry.register()` decorator.

### Agent Execution Flow

```
AgentExecutor.execute()
    ├── If starts with "llm:" → _execute_llm_profile()
    └── Else → _execute_agent()
                 ├── If config type == "llm" → _execute_llm_agent()
                 └── Else → _execute_registered_agent()
                              └── _execute_agent_with_tools()  # Tool loop
```

---

## Tool Detection & Execution

### Pattern-Based Detection

**Location:** `nodes/tool_detection.py` → `ToolDetector`

Detects tool calls from natural language patterns in text:

| Pattern | Detected Tool |
|---------|---------------|
| `fix @file.py` | `read_file` |
| `read file.py` | `read_file` |
| `write X to file.py` | `write_file` |
| `move A to B` | `move_file` |
| `copy A to B` | `copy_file` |
| `delete file.py` | `delete_file` |
| `replace X with Y in file` | `replace_in_file` |

### Non-LLM Fast Path

**Location:** `nodes/non_llm_tools.py` → `run_non_llm_tools()`

Skips LLM entirely for:
- Greetings ("hi", "hello", "thanks")
- Explain requests ("what does X.py do")
- File operations detected by `_TOOL_DETECTOR`

---

## Verification & Retry Loop

### Trial-Error Loop

When a "fix task" is detected, the system implements:

```
1. Agent generates fix attempt
2. Review node runs verification (python main.py)
3. If output matches expected.txt → END
4. If mismatch → Build error message, set next_action="continue"
5. Loop back to plan node with error context
6. Repeat up to max_iterations (default 10)
```

### Local Repair Fallback

**Location:** `nodes/execution.py:544` → `_attempt_local_repair()`

Pattern-based mutations tried before looping:
1. `range(start, end)` → `range(start, end + 1)` (off-by-one)
2. Remove `if X: continue` guards
3. Remove `/100` divisions (discount calculation fix)

Also tries combinations of mutations.

### Max Iterations

Controlled by `config.runtime.max_iterations` (default 10).

---

## Configuration

### Key Config Paths

| Config | Purpose |
|--------|---------|
| `agents.preferences.default_agent` | Fallback agent |
| `agents.configs.<name>` | Agent-specific config |
| `llm.default_profile` | Default LLM for planning |
| `runtime.max_iterations` | Max retry loops |
| `runtime.detect_tool_calls_from_text` | Enable pattern detection |
| `planning.max_retry_attempts` | LLM call retries |

### Agent Reference Formats

```yaml
# In prompts/decisions:
"agent:ollama"           # Registered agent
"agent:claude_code"      # Registered agent
"llm:groq-qwen3-32b"     # LLM profile
"agent:groq-qwen3-32b"   # Config-defined agent
```

---

## Debugging Tips

### Enable Verbose Mode
Set `verbose: true` in state or pass `--verbose` to CLI.

### Check Debug Logs
Challenges write to `codur_debug.log` in the challenge directory.

### Common Issues

1. **Recursion limit hit** → Increase in `main_graph.py` or reduce iterations
2. **Agent not found** → Check `AgentRegistry.list_agents()` and config
3. **Tool not executed** → Check pattern in `tool_detection.py`
4. **Verification always fails** → Check `expected.txt` format and encoding

---

## Extending the System

### Adding a New Agent

1. Create class in `codur/agents/new_agent.py`
2. Inherit from `BaseAgent`
3. Implement `execute()` and `aexecute()`
4. Register with `@AgentRegistry.register("new_agent")`
5. Import in `nodes/execution.py` to trigger registration

### Adding a New Tool

1. Implement in `codur/tools/`
2. Add to `tool_map` in `nodes/tools.py`
3. Add detection pattern in `nodes/tool_detection.py`
4. Export in `codur/tools/__init__.py`

### Modifying the Graph

Edit `main_graph.py`:
- Add nodes with `workflow.add_node()`
- Add edges with `workflow.add_edge()` or `workflow.add_conditional_edges()`
- Update routing functions in `nodes/routing.py`

---

*Last updated: 2025-12-22*
