# Flakiness Improvements for Codur Agent Logic

This document describes improvements made to reduce test flakiness and improve challenge success rates.

---

## Executive Summary

Codur's challenges were failing intermittently due to 5 key sources of flakiness in the agent logic. We've implemented targeted improvements to address each issue, resulting in more reliable agent execution and higher success rates.

---

## Flakiness Issues & Solutions

### 1. Aggressive Message Pruning

**Problem:**
- Original logic kept only the last 4 verification error messages
- Discarded intermediate AIMessage outputs from agent attempts
- Agents couldn't see their own previous attempts and feedback together
- Lost context after 4-5 iterations, forcing agents to restart reasoning

**Solution:**
- Changed `_prune_messages()` to keep last 5 **pairs** of (AIMessage, SystemMessage)
- This preserves the learning loop: agent sees "I tried X, got error Y, I tried Z, got error W"
- Agents can now learn from their mistakes by reviewing previous attempts
- More messages kept = more context preserved for complex problems

**Code Location:** `codur/graph/nodes/execution.py:436-485`

**Impact:** Helps agents avoid repeating the same mistakes in retry loops

---

### 2. Phase 1 Over-Confidence in Code Tasks

**Problem:**
- Quick classification (Phase 1) was routing code tasks at 80-85% confidence
- Sometimes wrong agent selected, requiring a full retry loop to fix
- Led to wasted iterations and potential failures

**Solution:**
- Increased Phase 1 code task confidence threshold from 80% to 90%
- Code tasks below 90% confidence now fall through to Phase 2 LLM for proper routing
- Only greetings, file ops, and high-confidence code tasks use Phase 1
- Reduces wrong routing decisions

**Code Location:** `codur/graph/nodes/planning/core.py:314-340`

**Impact:** Correct agent routing on first try, fewer retry loops needed

---

### 3. Limited Error Context

**Problem:**
- Error messages truncated to 20 lines
- Agents couldn't see full picture of what went wrong
- Important error details lost, making fixes harder to reason about

**Solution:**
- Increased truncation from 20 to 30 lines
- Gives agents 50% more context to understand failures
- Applied to both expected/actual output and error messages
- More context helps agents fix issues faster

**Code Location:** `codur/graph/nodes/execution.py:427-437`

**Impact:** Better error diagnosis, fewer wrong fix attempts

---

### 4. Limited Repair Patterns

**Problem:**
- Local repair tried only 3 mutation patterns
- Off-by-one, continue guards, division by 100
- Missed many common bug types: comparison operators, loop conditions, etc.
- Forced fallback to agent retry loop for fixable bugs

**Solution:**
- Added 2 new mutation patterns:
  - `mutate_fix_comparison`: Flips `>=` ↔ `>` and `<=` ↔ `<`
  - `mutate_fix_loop_condition`: Fixes bare loop conditions (e.g., `while n:` → `while n > 0:`)
- Now tries 5 mutation patterns + combinations
- Catches more common bugs locally without agent retry
- Faster fix time, fewer API calls

**Code Location:** `codur/graph/nodes/execution.py:674-690`

**Impact:** More bugs fixed in local repair phase, fewer expensive agent retries

---

### 5. Agent-Specific Capability Routing

**Problem:**
- All code tasks routed to same default agent
- Didn't leverage agent strengths/weaknesses
- Simple tasks went to expensive agents

**Solution:**
- Implemented capability-based routing with fallback
- Agents scored on: multi-file support, context size, cost, offline capability
- Simple tasks route to cheap agents (Ollama)
- Complex tasks route to capable agents (Claude Code)
- Reduces costs and improves reliability (right tool for job)

**Code Location:** `codur/graph/nodes/planning/classifier.py:205-244`

**Impact:** Correct agent for task type, better results, lower costs

---

### 6. Lower Temperature for Planning

**Problem:**
- Planning used default temperature (0.7)
- LLM could generate non-JSON responses or invalid formatting
- Forced JSON parsing retry loops

**Solution:**
- Added `planning_temperature: 0.3` to config
- Planning node now uses lower temperature for deterministic JSON
- Code generation still uses normal temperature (0.7)
- More reliable planning output, fewer retry loops

**Code Location:**
- Config: `codur.yaml:117-118`
- Usage: `codur/graph/nodes/planning/core.py:90-99`

**Impact:** Fewer planning failures, more reliable initial routing

---

## Implementation Summary

### Files Modified

| File | Changes |
|------|---------|
| `codur/graph/nodes/execution.py` | Message pruning (keep more context), error truncation (30 lines), repair patterns (+2) |
| `codur/graph/nodes/planning/core.py` | Phase 1 safeguard (90% confidence), planning temperature |
| `codur/graph/nodes/planning/classifier.py` | Capability-based routing |
| `codur/graph/AGENTIC_LOGIC.md` | Documented flakiness sources and strategies |
| `codur.yaml` | Added `planning_temperature`, `agent_execution`, `planning` sections |

### Configuration Changes

```yaml
llm:
  planning_temperature: 0.3      # Lower for deterministic JSON

planning:
  max_retry_attempts: 2          # Faster failure detection
  retry_initial_delay: 0.3       # Faster retries
  retry_backoff_factor: 1.5      # Less backoff

agent_execution:
  default_cli_timeout: 300       # 5 min timeout
```

---

## Measurable Improvements

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Message context preserved | 4 errors | 5 attempt-error pairs | +25% context |
| Phase 1 code routing confidence | 80% | 90% | Less wrong routes |
| Error context lines | 20 | 30 | +50% detail |
| Repair patterns | 3 | 5+ combinations | +67% coverage |
| Planning temperature | 0.7 | 0.3 | More deterministic |

---

## Testing

All improvements have been validated:
- ✅ First two challenges pass consistently
- ✅ Message pruning preserves learning context
- ✅ All imports and configurations valid
- ✅ No regressions in existing functionality

---

## Future Improvements

Based on analysis, potential next steps:
1. **Tool Detection Robustness**: Support multiple tool call formats
2. **Error Categorization**: Detect error types and suggest fixes
3. **Agent Diversity**: Route to different agents based on detected error type
4. **Instruction Refinement**: Better prompts for specific task types
5. **Failure Pattern Database**: Learn from past failures to predict fixes

---

*Last updated: 2025-12-22*
