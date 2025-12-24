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

**Phase Summary:**
- **Phase 0** (pattern_plan): Pattern-based classification and routing (NO LLM)
- **Phase 1** (llm_pre_plan): Fast LLM classification (**ENABLED BY DEFAULT**, uses Groq)
- **Phase 2** (llm_plan): Full LLM planning with context-aware prompts (uses Groq)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           THREE-PHASE GRAPH FLOW                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐          │
│   │  PATTERN PLAN    │►│ LLM PRE-PLAN    │►│    LLM PLAN     │          │
│   │ (No LLM calls)   │ │ (Fast LLM)      │ │ (Full planning) │          │
│   │ • Instant        │ │ • Experimental  │ │ • Groq-based    │          │
│   │ • Classification │ │ • Config-gated  │ │ • Complex tools │          │
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
│   │  to LLM PLAN      │                                                  │
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
| `nodes/planning/core.py` | PlanningOrchestrator - Phase 2 LLM planning |
| `nodes/planning/strategies/` | TaskStrategy package - task-specific strategies |
| `nodes/planning/tool_analysis.py` | Helpers for analyzing tool outputs (discovery) |
| `nodes/execution.py` | AgentExecutor, delegate_node, execute_node, review_node |
| `nodes/tools.py` | Tool execution node |
| `nodes/coding.py` | codur-coding dedicated execution node |
| `nodes/tool_detection.py` | Pattern-based tool call detection from text |
| `nodes/non_llm_tools.py` | Fast-path tool detection without LLM |

---

## Graph Flow

### Entry Point
The graph starts at the **pattern_plan** node (Phase 0, pattern-based). See `main_graph.py:95`.

### Nodes

**Planning Phase (Phases 0-2):**
1. **pattern_plan** → Pattern-based detection (Phase 0, NO LLM)
   - Instant resolution for trivial cases (greetings, basic file ops)
   - Pattern classification with task-specific strategies
2. **llm_pre_plan** → Fast LLM classification (Phase 1, **ENABLED BY DEFAULT**)
   - Uses Groq for fast, smart classification (temp=0.2)
   - Handles nuanced cases that patterns miss (e.g., "hey fix main.py" → CODE_FIX)
   - Can be disabled with `config.planning.use_llm_pre_plan=False` for pattern-only mode
3. **llm_plan** → Full LLM planning for uncertain cases (Phase 2, uses Groq)

**Execution Phase:**
4. **delegate** → Routes task to selected agent
5. **execute** → Runs the agent with tool-using loop
6. **tool** → Executes filesystem/MCP tools directly
7. **coding** → Runs the codur-coding agent with challenge + optional context
7. **review** → Verifies result, decides to loop or end

### Edges

```python
# Phase 0 → Phase 1 (pattern_plan → llm_pre_plan)
pattern_plan → llm_pre_plan      # if next_action == "continue_to_llm_pre_plan"
pattern_plan → delegate           # if next_action == "delegate" and selected_agent != "agent:codur-coding"
pattern_plan → coding             # if next_action == "delegate" and selected_agent == "agent:codur-coding"
pattern_plan → tool               # if next_action == "tool" (fast-path resolved)
pattern_plan → END                # if next_action == "end" (fast-path resolved)

# Phase 1 → Phase 2 (llm_pre_plan → llm_plan)
llm_pre_plan → llm_plan               # if next_action == "continue_to_llm_plan"
llm_pre_plan → delegate               # if next_action == "delegate" and selected_agent != "agent:codur-coding" (high confidence)
llm_pre_plan → coding                 # if next_action == "delegate" and selected_agent == "agent:codur-coding" (high confidence)
llm_pre_plan → tool                   # if next_action == "tool" (high confidence)
llm_pre_plan → END                    # if next_action == "end" (high confidence)

# Phase 2 → Execution (llm_plan → delegate/tool/END)
llm_plan → delegate                   # if next_action == "delegate" and selected_agent != "agent:codur-coding"
llm_plan → coding                     # if next_action == "delegate" and selected_agent == "agent:codur-coding"
llm_plan → tool                       # if next_action == "tool"
llm_plan → END                        # if next_action == "end"

# Execution phase
delegate → execute                    # Fixed edge (routes to agent executor)
coding → review                       # Fixed edge (routes directly to verification, no agent executor)
tool → review                         # Fixed edge (routes directly to verification, no tool executor wrapper)
execute → review                      # Fixed edge (routes from agent executor)

# Retry loop (review → back to Phase 2 for retries, skipping Phases 0-1)
review → llm_plan                     # if next_action == "continue" (retry - already classified)
review → END                          # if next_action == "end" (done)
```

**Note on Routing:** When the planner returns `action: "delegate"` with `selected_agent: "agent:codur-coding"`, the router checks the agent name and routes to the "coding" node instead of the standard "delegate" node. This is a special route for coding-optimized LLM execution without the general agent executor wrapper.

### Recursion Limit

The graph is compiled with `recursion_limit=350` to support:
- Initial planning phase: ~6 nodes (pattern_plan → llm_pre_plan → llm_plan → delegate/tool/coding → execute/review)
- Up to 10 max review iterations with retry loop
- 5 tool iterations per agent execution
- Safety margin for complex tasks with multiple retries

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
    tool_calls: list[dict]           # Tool calls requested by planner/hints
    verbose: bool                    # Enable verbose logging
    config: CodurConfig              # Runtime configuration
```

### Message Accumulation

The `messages` field uses `Annotated[Sequence[BaseMessage], operator.add]` which means messages are **accumulated** across nodes, not replaced.

### Message Pruning

To prevent context explosion during retry loops, `_prune_messages()` in `execution.py:417` keeps:
- Original HumanMessage (the task)
- Last 8 verification error messages (and recent agent attempts)
- Discards older intermediate messages

**Note:** Message pruning can cause flakiness if it discards important context. Consider:
- Keeping more recent error messages (6-8 instead of 4) for complex tasks
- Including AIMessage outputs from recent attempts (agents learn from their own attempts)
- Preserving agent reasoning if available

---

## Node Implementations

### 1. Plan Nodes

#### pattern_plan (Phase 0)
**Location:** `nodes/planning/core.py:37`

**No LLM calls** - Pure pattern matching combining:
1. **Instant resolution** via `run_non_llm_tools()` for trivial cases (greetings, basic file ops)
2. **Classification** via `quick_classify()` for task type detection (pattern-based, no LLM)
3. **Strategy execution** from `strategies/` package based on TaskType
   - Resolve simple tasks immediately (greetings, file operations, web search)
   - Trigger discovery tools (e.g., `list_files`, `read_file`)
   - **Conservative**: Does NOT delegate code tasks (passes to Phase 2)

**Design Philosophy**: Phase 0 focuses on **discovery and simple tasks only**. For code fix/generation/refactor tasks, it gathers context (reads files) but defers routing decisions to Phase 2 LLM planning. This prevents premature delegation of complex tasks.

If no pattern matches, passes to Phase 1 (or Phase 2 if LLM pre-plan disabled).

#### llm_pre_plan (Phase 1)
**Location:** `nodes/planning/core.py:106`

**Smart LLM-based classification** (ENABLED BY DEFAULT):
- Uses Groq for fast classification with lightweight prompt
- Handles nuanced cases that pattern matching misses:
  - "hey fix main.py" → CODE_FIX (not GREETING)
  - "remove debug prints from app.py" → CODE_FIX (not FILE_OPERATION)
  - "What does app.py do? Please fix the bug." → CODE_FIX (not EXPLANATION)
- Returns: `{task_type, confidence, detected_files, suggested_action, reasoning}`
- High confidence (≥0.8) → may route or trigger discovery
- Low confidence → passes to Phase 2

**Flow:**
1. Call Groq with classification prompt (JSON mode, temp=0.2)
2. Parse classification result with context understanding
3. If confident → route; else → pass to Phase 2

**Disabling (not recommended):**
```yaml
planning:
  use_llm_pre_plan: false  # Fall back to pattern-only (less accurate)
```

#### llm_plan (Phase 2)
**Location:** `nodes/planning/core.py` → `PlanningOrchestrator.llm_plan()`

Full LLM planning using context-aware prompts. Handles complex routing, compound tool calls, and retries based on verification errors. Uses Groq (default profile).

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

### 4. Coding Node

**Location:** `nodes/coding.py` → `coding_node()`

**Purpose:** Run the codur-coding agent with a structured coding prompt (JSON mode).

**Dynamic Path Handling:**
The coding node does **not** hardcode `main.py`. It determines the target path by:
1. Checking current `tool_calls` in the LLM response.
2. Checking `tool_calls` in the `AgentState` (populated by Phase 1 hints or Phase 2 planner).
3. Failing gracefully if no path can be determined.

---

### 5. Tool Node

**Location:** `nodes/tools.py:56` → `tool_node()`

**Purpose:** Execute filesystem and MCP tools directly (not via agent).

**Supported Tools:**
- File operations: `read_file`, `write_file`, `append_file`, `delete_file`, `copy_file`, `move_file`
- Search: `search_files`, `grep_files`, `list_files`, `list_dirs`, `file_tree`
- Structured data: `read_json`, `write_json`, `read_yaml`, `write_yaml`, `read_ini`, `write_ini`
- Git: `git_status`, `git_diff`, `git_log`, `git_stage_files`, `git_commit`
- MCP: `list_mcp_tools`, `call_mcp_tool`, `read_mcp_resource`
- Web: `fetch_webpage`, `duckduckgo_search`
- Python AST: `python_ast_graph`, `python_ast_outline`, `python_ast_dependencies`, `python_ast_dependencies_multifile`

**Automatic Augmentation:**
When `read_file` targets a `.py` file, the tool node automatically adds a `python_ast_dependencies` call for the same path.
When `list_files` returns 1-5 Python files, the tool node adds `python_ast_dependencies_multifile` for those paths.

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
   - Increment iteration counter
4. If verification passes or not a fix task → set `next_action: "end"`

**Retry Behavior:**
- On `next_action: "continue"`: Routes back to `llm_plan` node (Phase 2)
  - Skips Phases 0-1 (pattern_plan, llm_pre_plan) since task is already classified
  - Provides error context to planner for targeted fix
  - Continues until max_iterations reached or verification passes
- On `next_action: "end"`: Routes to graph END

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
| ````json ... ```` | JSON-formatted tool calls |

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
5. Loop back to llm_plan node (Phase 2) with error context
   - Skips Phases 0-1 since task is already classified
   - Prunes older messages to prevent context explosion
   - Provides structured error details for planner
6. Repeat up to max_iterations (default 10)
```

### Local Repair Fallback

**Location:** `nodes/execution.py:544` → `_attempt_local_repair()`

Pattern-based mutations tried before looping:
1. `range(start, end)` → `range(start, end + 1)` (off-by-one)
2. Remove `if X: continue` guards
3. Remove `/100` divisions (discount calculation fix)
4. Fix list access `[i]` → `[i-1]`
5. Add missing f-string prefix

Also tries combinations of mutations.

### Max Iterations

Controlled by `config.runtime.max_iterations` (default 10).

---

## Configuration

### Key Config Paths

| Config | Purpose | Default |
|--------|---------|---------|
| `agents.preferences.default_agent` | Fallback agent | - |
| `agents.configs.<name>` | Agent-specific config | - |
| `llm.default_profile` | Default LLM for planning (Groq) | groq-qwen3-32b |
| `runtime.max_iterations` | Max retry loops | 10 |
| `runtime.detect_tool_calls_from_text` | Enable pattern detection | True |
| `planning.max_retry_attempts` | LLM call retries | 3 |
| **`planning.use_llm_pre_plan`** | **Enable Phase 1 LLM classification** | **True** ✅ |

### LLM Pre-Plan (Phase 1) - ENABLED BY DEFAULT

Phase 1 uses Groq for smart task classification, enabled by default for better accuracy.

**Benefits:**
- Handles nuanced cases that pattern matching misses
- Understands context and intent, not just keywords
- More flexible for novel task types
- Better file path detection via LLM understanding

**Performance:**
- Adds ~200-500ms latency (Groq is fast)
- Small cost increase (Groq is cheap)
- Worth it for significantly better routing accuracy

**Disabling (not recommended):**
```yaml
planning:
  use_llm_pre_plan: false  # Fall back to pattern-only mode (less accurate)
```

**Why it's default:** Pattern-based classification has known misroutes (see `test_classifier_misroutes.py`). LLM classification solves these edge cases.

### Codur Coding Agent (agent:codur-coding)

The `codur-coding` agent profile is a coding-optimized LLM configuration intended for coding challenges with optional extra context. It uses a system prompt that:
- Solves the task end-to-end from the provided problem statement
- Restates assumptions briefly
- Produces correct, efficient code with edge cases
- Asks a single concise clarification only if the task is ambiguous

**JSON Output Requirement (enforced):**
The coding node runs the LLM in JSON mode and expects a valid JSON object with tool calls:
```json
{
  "thought": "brief reasoning",
  "tool_calls": [
    {
      "tool": "replace_function",
      "args": {
        "path": "main.py",
        "function_name": "name_of_function",
        "new_code": "def name_of_function(...):\\n    ..."
      }
    }
  ]
}
```

**Available Coding Tools:**
- `replace_function(path, function_name, new_code)`
- `replace_class(path, class_name, new_code)`
- `replace_method(path, class_name, method_name, new_code)`
- `replace_file_content(path, new_code)`
- `inject_function(path, new_code, function_name?)`

**Fallback/Defaults:**
- If `tool_calls` is missing but `code` exists, the node treats it as a full file replacement of `main.py`.
- If a tool call omits `path`, it defaults to `main.py`.

Use this profile when routing tasks that are primarily coding challenges and include supplemental context that should be considered during solution.

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
2. **Phase 0 Over-Confidence** - Pattern classification at 80% confidence might route complex tasks incorrectly
3. **Pattern-Based Tool Detection** - Agents might format tool calls differently than expected
4. **Limited Error Information** - Truncated error messages don't show full context of what failed
5. **Limited Repair Patterns** - Only 3 mutation patterns covers a tiny fraction of possible bugs
6. **Pattern Rigidity** - Phase 0 patterns are hardcoded and miss novel task variations (can be mitigated by enabling Phase 1 LLM pre-plan)

### Strategies for Improving Reliability

#### Message Pruning
- Keep more error messages (6-8 instead of 4) to preserve agent learning context
- Include recent AIMessage outputs so agents see their own attempts
- Only prune very old messages (> 5 iterations ago)

#### Phase 0/1 Safety
- Phase 0 patterns work well for obvious cases but can be rigid
- Consider enabling Phase 1 LLM pre-plan (`use_llm_pre_plan: true`) for better generalization
- Pattern confidence threshold (0.8) is reasonable but may need tuning per task type
- For novel task variations, Phase 1 LLM provides better adaptability than hardcoded patterns

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

## Changelog

### 2025-12-24
- **Phase 1 LLM classification ENABLED BY DEFAULT**: Smart classification instead of pattern-only
  - Changed `use_llm_pre_plan: false` → `true` in default config
  - Fixes known misroutes: "hey fix main.py", "remove debug prints", etc.
  - Uses Groq for fast LLM-based classification (~200-500ms)
  - Pattern-only mode still available but not recommended
- **Phase 0 conservative strategy**: Made Phase 0 more conservative to reduce brittleness
  - Code fix/generation/refactor strategies now do **discovery only**, not delegation
  - Phase 0 reads files and gathers context, Phase 2 makes routing decisions
  - Prevents premature delegation of complex tasks with insufficient context
  - Matches original safer behavior where Phase 1 did discovery before Phase 2 routing

### 2025-12-23
- **Phase 0 refactor**: Merged `textual_pre_plan` and old `llm_pre_plan` into unified `pattern_plan`
  - Phase 0 now combines instant resolution + pattern classification (NO LLM calls)
  - Clarified naming: Phase 0 = pattern-based, Phase 1 = LLM-based (experimental)
- **Phase 1 enhancement**: Added true LLM-based classification (config-gated)
  - New config flag: `planning.use_llm_pre_plan` (default: False)
  - Uses Groq for fast classification when enabled
  - More flexible than patterns for novel task types
- **Documentation cleanup**: Fixed misleading "LLM Pre-Plan" terminology
  - Updated all diagrams and descriptions to reflect actual LLM usage
  - Added experimental flag documentation

---

*Last updated: 2025-12-24*
