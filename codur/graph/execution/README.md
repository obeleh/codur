# Execution Module

This module implements the execution, delegation, and review logic for the Codur agent graph. It handles routing tasks to appropriate agents, executing them, and verifying results.

## Architecture

The execution module is organized into focused, single-responsibility modules:

### Core Components

- **`delegate.py`** - Routes tasks to the appropriate agent based on planning decisions
- **`execute.py`** - Wrapper node that instantiates and runs the AgentExecutor
- **`review.py`** - Verifies fix results, routes retries, and decides on next actions
- **`agent_executor.py`** - Core execution engine with tool-loop support

### Supporting Logic

- **`verification.py`** - Runs tests and validates outputs against expected results
- **`repair.py`** - Last-resort mutation-based repair for common coding errors

## Execution Flow

```
delegate_node → execute_node → [agent execution] → review_node
                    ↓
            AgentExecutor (with tool loop)
                    ↓
         [Tool detection & execution]
                    ↓
              review_node decides:
         ├─ Verification passed? → END
         ├─ Verification failed? → retry or replan
         ├─ Stuck in loop? → END
         └─ Tool result only? → continue to planning
```

## Module Responsibilities

### `delegate_node`
- Routes to the agent selected by the planning phase
- Falls back to configured default agent if needed
- Returns: `{"agent_outcome": {"agent": name, "status": "delegated"}}`

### `execute_node`
- Creates an AgentExecutor instance
- Delegates all execution logic to AgentExecutor
- Returns: `{"agent_outcome": {...}, "llm_calls": count}`

### `AgentExecutor`
- Main execution engine for agents
- Implements tool-using loop (max 5 iterations by default)
- Supports three agent types:
  1. LLM profiles (direct LLM calls)
  2. LLM agents (agents configured with LLM models)
  3. Registered agents (custom agent implementations)
- Tool loop:
  1. Agent generates response
  2. Detector checks for tool calls
  3. Tools are executed and results fed back
  4. Loop continues until no more tool calls or max iterations

### `review_node`
- Checks if results are acceptable
- Detects fix/debug tasks by keywords in original message
- Verification logic:
  1. If fix task: runs verification (python main.py or app.py)
  2. If output mismatch: compares with expected.txt
  3. If same error repeats: agent is stuck, exit
  4. If different error: retry with agent or attempt repair
  5. If success: accept result
- Routing decisions:
  - Tool result only → continue to planning
  - Verification passed → END
  - Repeated error → END
  - Verification failed → retry or route back to planning

### `_verify_fix` (in verification.py)
- Runs Python entry point (main.py or app.py)
- Streaming verification with early exit on mismatch
- Compares output against expected.txt if available
- Returns: `{"success": bool, "message": str, ...diagnostic_info}`

### `_attempt_local_repair` (in repair.py)
- Last-resort mutation-based repair
- Tries 7 common mutation patterns:
  - Range boundary fixes (off-by-one)
  - Continue guard removal
  - Division by 100 removal
  - Comparison operator flipping
  - Loop condition fixes
  - F-string prefix addition
  - List access fixes
- Tests mutations in parallel (4 workers)
- Returns: `{"success": bool, "message": str}`

## Integration Points

### State Management
All nodes receive and return `AgentState` which includes:
- Message history
- Tool calls and results
- Selected agent
- Iteration counters
- Error tracking

### Tool Execution
- AgentExecutor uses centralized `execute_tool_calls` from `codur/graph/tool_executor.py`
- Tool detection via `_TOOL_DETECTOR` for JSON and plain-text tool calls
- Tool results are fed back to agents for continuation

### Configuration
- Uses `CodurConfig` for runtime settings
- Respects `config.runtime.max_iterations` for verification retries
- Supports multiple LLM profiles for fallback execution

## Key Design Decisions

1. **Separate Modules**: Each node and major function gets its own file for clarity
2. **Tool Loop in AgentExecutor**: Agents can call tools and react to results
3. **Streaming Verification**: Early exit on output mismatch saves time
4. **Local Repair as Last Resort**: Attempts simple fixes before giving up
5. **Error Tracking**: Detects when agent is stuck and exits early

## Testing

All modules can be tested independently:
- Import specific functions: `from codur.graph.execution import delegate_node`
- AgentExecutor can be mocked for testing tool loops
- Verification can be tested with mock entry points
- Repair mutations can be unit tested separately

## See Also

- `AGENTIC_LOGIC.md` for the full planning and execution flow
- `CODING.md` for the coding agent tool loop specifics
- `codur/tools/README.md` for tool registry and authoring
- `codur/graph/planning/` for the planning phase logic
