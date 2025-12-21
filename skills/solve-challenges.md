# Solve Challenges Skill

This skill defines how to run and solve coding challenges to improve Codur's agentic capabilities.

## Goal

Make Codur smarter at solving challenges through improved agentic workflows, NOT by hardcoding solutions or adding Python logic that bypasses the LLM.

## Challenge Structure

Challenges are located in `challenges/` directory. Each challenge has:
- `prompt.txt` - The instruction sent to Codur
- `main.py` - The executable harness with a bug/issue
- `expected.txt` - The exact expected stdout from running `main.py`

## How to Run Challenges

```bash
# Run all challenges
pytest tests/test_challenges.py -v

# Run with verbose output to see what Codur is doing
pytest tests/test_challenges.py -v -s

# Run a single challenge manually
cd challenges/01-fix-off-by-onerror
python -m codur.cli --command "$(cat prompt.txt)" --raw --verbose --config ../../codur.yaml
python main.py  # Check if it works
```

## Success Criteria

A challenge is solved when:
1. Codur reads the prompt from `prompt.txt`
2. Codur autonomously fixes the bug in `main.py`
3. Running `python main.py` produces output matching `expected.txt`
4. **No hardcoded solutions** - the fix must come from intelligent agent behavior

## Trial-Error Loop Principle

Based on `skills/trial-error-loop.md`, agents should:
1. Run a CLI command (e.g., `python main.py`)
2. If it fails, inspect the error output
3. Make targeted file edits to resolve the error
4. Run the same CLI command again to verify the fix
5. Repeat this loop until the command succeeds
6. If you see connection errors to ollama. There must be a config issue. Halt and ask the user to fix it.
7. Do not modify codur.yaml config

## Current Issues to Fix

When solving challenges, focus on improving these areas:

### 1. Remove Hardcoded Solutions
❌ **Bad**: Adding examples like `"Fix bug in @main.py" → replace "range(start, end)" with "range(start, end + 1)"`
✅ **Good**: Making the planner delegate to an agent that can analyze and fix bugs

### 2. Give Agents Tool Access
❌ **Bad**: LLM agents that only generate text without ability to read/write files
✅ **Good**: Agents that can use tools like `read_file`, `write_file`, `bash` to implement trial-error loops

### 3. Implement Verification Loops
❌ **Bad**: Review node that accepts all results without testing
✅ **Good**: Review node that runs tests and loops back if they fail

### 4. Smart Agent Routing
❌ **Bad**: Always using the same agent regardless of task type
✅ **Good**: Routing "fix bug" tasks to agents capable of iteration and tool usage

### 5. Acceptable:

 - "Fix bug in @main.py" → {{"action": "delegate", "agent": "{default_agent}", "reasoning": "bug fix requires analysis", ...}}

## Workflow for Solving Challenges

1. **Run the test**: `pytest tests/test_challenges.py -v`
2. **Analyze failures**: Look at what went wrong
3. **Identify root cause**: Is it:
   - Hardcoded logic in prompts?
   - Agent lacking tool access?
   - Missing verification loop?
   - Poor agent routing?
4. **Improve agentic behavior**:
   - Update planning prompts to be smarter (without hardcoding)
   - Add tool access to agents
   - Implement trial-error loops in review node
   - Improve agent routing logic
5. **Test again**: Verify the fix works
6. **Ensure generalization**: The fix should work for similar challenges, not just this one

## Key Files to Modify

When improving Codur's challenge-solving abilities, these files are key:

- `codur/graph/nodes/planning.py` - Planning logic and routing decisions
- `codur/graph/nodes/execution.py` - Agent execution and review logic
- `codur/graph/nodes/non_llm_tools.py` - Tool detection and execution
- `codur/agents/*.py` - Individual agent implementations
- `codur/config.yaml` - Agent configuration and routing rules

## Anti-Patterns to Avoid

1. **Hardcoding specific fixes** in examples or tool detection
2. **Adding Python logic** that solves the problem without using an LLM
3. **Skipping verification** - always test that fixes actually work
4. **One-shot solutions** - implement loops for iterative improvement
5. **Agent-specific hardcoding** - make routing dynamic, not fixed
6. **Hacks** - remove recursion limits

## Success Indicators

You've successfully improved Codur when:
- ✅ All challenge tests pass
- ✅ No hardcoded solutions in the codebase
- ✅ Agents can iterate and verify their own work
- ✅ The system generalizes to new, unseen challenges
- ✅ Verbose output shows intelligent agent decision-making

## Example: Good vs Bad Improvements

### Bad Approach
```python
# In planning.py - HARDCODED SOLUTION
if "off-by-one" in task.lower():
    return {"tool_calls": [{"tool": "replace_in_file", "args": {
        "pattern": "range(start, end)",
        "replacement": "range(start, end + 1)"
    }}]}
```

### Good Approach
```python
# In planning.py - SMART DELEGATION
if "fix" in task.lower() and ".py" in task:
    # Delegate to an agent that can read, analyze, fix, and verify
    return {"action": "delegate", "agent": default_agent}

# In execution.py - GIVE AGENTS TOOLS
llm_with_tools = llm.bind_tools([read_file, write_file, bash])
# Agent can now iterate with trial-error loop
```

## Next Steps

After solving current challenges:
1. Add more diverse challenges to test generalization
2. Measure success rate across all challenges
3. Identify patterns in failures
4. Continue improving agentic workflows
5. Document learnings in this skill
