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

Codur uses **LangGraph** to orchestrate an agentic workflow. The system follows a three-phase planning architecture with automatic retry loops for fix tasks.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           THREE-PHASE GRAPH FLOW                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐          │
│   │ TEXTUAL PRE-PLAN │►│  LLM PRE-PLAN   │►│    LLM PLAN     │          │
│   │ (Fast patterns)  │ │ (Quick classify) │ │ (Full planning) │          │
│   └────────┬─────────┘ └────────┬─────────┘ └────────┬────────┘          │
│            │                    │                    │                    │
│   ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐          │
│   │   DELEGATE      │  │   DELEGATE      │  │   DELEGATE      │          │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│            │                    │                    │                    │
│   ┌────────▼──────────────────────────────────────────▼──────┐           │
│   │                      EXECUTE → REVIEW                    │           │
│   └────────┬─────────────────────────────────────────┬───────┘           │
│            │                                         │                    │
│   ┌────────▼────────┐                        ┌──────▼──────┐             │
│   │   TOOL          │                        │    END      │             │
│   └────────┬────────┘                        └─────────────┘             │
│            │                                                              │
│   ┌────────▼──────────┐                                                  │
│   │ continue → back   │                                                  │
│   │ to TEXTUAL PRE    │                                                  │
│   └───────────────────┘                                                  │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
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
The graph starts at the **textual_pre_plan** node (Phase 0). See `main_graph.py:91`.

### Nodes

**Planning Phase (Phases 0-2):**
1. **textual_pre_plan** → Fast pattern-based detection (Phase 0)
2. **llm_pre_plan** → Quick LLM classification for high-confidence cases (Phase 1)
3. **llm_plan** → Full LLM planning for uncertain cases (Phase 2)

**Execution Phase:**
4. **delegate** → Routes task to selected agent
5. **execute** → Runs the agent with tool-using loop
6. **tool** → Executes filesystem/MCP tools directly
7. **review** → Verifies result, decides to loop or end

### Edges

```python
# Phase 0 → Phase 1 (textual_pre_plan → llm_pre_plan)
textual_pre_plan → llm_pre_plan      # if next_action == "continue_to_llm_pre_plan"
textual_pre_plan → delegate           # if next_action == "delegate" (fast-path resolved)
textual_pre_plan → tool               # if next_action == "tool" (fast-path resolved)
textual_pre_plan → END                # if next_action == "end" (fast-path resolved)

# Phase 1 → Phase 2 (llm_pre_plan → llm_plan)
llm_pre_plan → llm_plan               # if next_action == "continue_to_llm_plan"
llm_pre_plan → delegate               # if next_action == "delegate" (high confidence)
llm_pre_plan → tool                   # if next_action == "tool" (high confidence)
llm_pre_plan → END                    # if next_action == "end" (high confidence)

# Phase 2 → Execution (llm_plan → delegate/tool/END)
llm_plan → delegate                   # if next_action == "delegate"
llm_plan → tool                       # if next_action == "tool"
llm_plan → END                        # if next_action == "end"

# Execution phase
delegate → execute                    # Fixed edge
execute → review                      # Fixed edge
tool → review                         # Fixed edge

# Retry loop (review → back to Phase 0)
review → textual_pre_plan             # if next_action == "continue" (retry)
review → END                          # if next_action == "end" (done)
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

**Note:** Message pruning can cause flakiness if it discards important context. Consider:
- Keeping more recent error messages (6-8 instead of 4) for complex tasks
- Including AIMessage outputs from recent attempts (agents learn from their own attempts)
- Preserving agent reasoning if available

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
    "action": "delegate" OR "tool" OR "respond" OR "done",
    "agent": "agent:ollama" OR "llm:groq-qwen3-32b" OR null,
    "reasoning": "brief explanation",
    "response": "only for greetings",
    "tool_calls": [{"tool": "read_file", "args": {"path": "..."}}]
}
```

**Returns:** `PlanNodeResult` with `next_action`, `selected_agent`, `tool_calls`, etc.

**Flakiness Note:** Phase 1 quick classification is used for high-confidence cases (≥80%), but can cause issues if it misclassifies complex tasks. Current implementation has safeguards but consider being more conservative with Phase 1 to avoid wrong routing.

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

## Flakiness & Reliability

### Common Sources of Flakiness

1. **Aggressive Message Pruning** - Only keeping last 4 error messages can lose context agents need to fix issues
2. **Phase 1 Over-Confidence** - Quick classification at 80% confidence might route complex tasks incorrectly
3. **Pattern-Based Tool Detection** - Agents might format tool calls differently than expected
4. **Limited Error Information** - Truncated error messages don't show full context of what failed
5. **Limited Repair Patterns** - Only 3 mutation patterns covers a tiny fraction of possible bugs

### Strategies for Improving Reliability

#### Message Pruning
- Keep more error messages (6-8 instead of 4) to preserve agent learning context
- Include recent AIMessage outputs so agents see their own attempts
- Only prune very old messages (> 5 iterations ago)

#### Phase 1 Safety
- Use Phase 1 only for truly obvious cases: greetings, simple file ops
- For code tasks, always use Phase 2 LLM to get proper routing
- Require 90%+ confidence for Phase 1 code task decisions

#### Error Message Quality
- Show more context (30 lines instead of 20) for error messages
- Include which attempts failed and why
- Show agent previous attempts so they don't repeat mistakes
- Format with clear structure: Expected → Actual → Error → Context

#### Tool Detection Robustness
- Support multiple tool call formats (agent might use backticks, code blocks, etc)
- Fall back to LLM-based tool detection if pattern matching fails
- Log what tool detection found for debugging

#### Better Repair Patterns
- Add pattern for `list comprehension` errors
- Add pattern for `string formatting` bugs
- Add pattern for `type mismatch` issues
- Support agent hints ("looks like off-by-one")

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
