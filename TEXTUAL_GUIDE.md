# Textual TUI Guide for Codur

## Why Textual?

**Textual** is a modern async TUI framework perfect for Codur because:

- ✅ **Fully async** - Built on asyncio, perfect for concurrent agent execution
- ✅ **Live updates** - Stream agent progress in real-time
- ✅ **Split interface** - Show logs + accept input simultaneously
- ✅ **Beautiful** - Modern terminal UI with colors and widgets
- ✅ **Interactive** - Users can guide agents mid-execution

## Installation

```bash
pip install textual
# or
pip install -e ".[dev]"
```

## Running the TUI

### Option 1: Via CLI command
```bash
codur tui
```

### Option 2: Standalone script
```bash
python -m codur.tui_standalone
```

### Option 3: Direct import
```python
from codur.tui import run_tui
run_tui()
```

## TUI Layout

```
┌─────────────────────────────────────────────────┐
│ Codur Agent TUI                         Ctrl+C  │
├─────────────────────────────────────────────────┤
│ Agent: Ollama | Step: Planning | Iteration: 3   │ ← Status Bar
├─────────────────────────────────────────────────┤
│                                                  │
│  16:30:45 Starting task: Write fibonacci        │ ← Log Pane
│  16:30:46 Step 1: Planning                      │   (70% height)
│  16:30:48 Step 2: Delegating to Ollama          │   Real-time logs
│  16:30:50 Step 3: Executing...                  │
│  16:30:52 Ollama generated 234 characters       │
│                                                  │
├─────────────────────────────────────────────────┤
│ Commands: :pause :resume :hint <text>           │ ← Input Pane
│ > :hint use recursion instead                   │   (30% height)
│                                                  │   User can type
└─────────────────────────────────────────────────┘   while running
```

## Features

### Real-Time Interaction

**While an agent is running, you can:**

1. **Type new tasks** - Queue additional work
2. **Pause execution** - Press `Ctrl+P` or type `:pause`
3. **Resume** - Press `Ctrl+R` or type `:resume`
4. **Provide hints** - Type `:hint use FastAPI instead`
5. **Switch agents** - Type `:set agent=codex`

### Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `:pause` | `Ctrl+P` | Pause agent execution |
| `:resume` | `Ctrl+R` | Resume agent |
| `:hint <text>` | - | Provide guidance mid-execution |
| `:set agent=<name>` | - | Switch to specific agent |
| `:help` | - | Show help |
| Quit | `Ctrl+C` | Exit TUI |
| Clear input | `Esc` | Clear input field |

### Status Bar

Shows real-time information:
- **Current agent**: Which agent is running (Ollama, Codex, etc.)
- **Current step**: Plan, Delegate, Execute, Review
- **Iteration count**: How many loops executed

### Log Pane

Displays:
- Timestamped messages
- Agent progress updates
- User commands
- Errors and warnings
- Color-coded by severity

### Input Pane

- Always accepts input (even while agent runs)
- Command history
- Syntax highlighting
- Auto-complete (future)

## Example Session

```
# Start TUI
$ codur tui

# User types task
> Write a Python function to calculate fibonacci

# Agent starts, logs appear:
16:30:45 Starting task: Write fibonacci
16:30:46 Agent: Ollama | Step: Planning
16:30:48 Step: Delegating to Ollama
16:30:50 Step: Executing...

# User provides guidance mid-execution
> :hint use dynamic programming approach

16:30:51 Hint added: use dynamic programming approach
16:30:52 Ollama adapting to hint...

# User pauses to review
> :pause

16:30:53 Agent paused

# Review output, then resume
> :resume

16:30:55 Agent resumed
16:30:57 Task completed!
```

## Architecture Integration

### Async Event Loop

The TUI runs in the same event loop as the LangGraph agent:

```python
async def run_agent(task: str):
    # Main agent task
    async for event in graph.astream_events(task):
        # Update UI with event
        self.log_message(event)

        # Check for user input
        if not self.user_queue.empty():
            guidance = await self.user_queue.get()
            # Inject into agent
            await graph.update_state(guidance)
```

### Concurrent Tasks

```
┌─────────────────┐
│  Textual App    │
│  Event Loop     │
└────────┬────────┘
         │
    ┌────┴────┬─────────┬──────────┐
    │         │         │          │
┌───▼───┐ ┌──▼───┐ ┌───▼────┐ ┌───▼────┐
│ Graph │ │ Input│ │ Status │ │  Log   │
│Worker │ │Worker│ │Updater │ │Streamer│
└───────┘ └──────┘ └────────┘ └────────┘
```

## Comparison: Textual vs Others

### Textual (Recommended)
✅ Async native
✅ Beautiful widgets
✅ Split-pane layouts
✅ Keyboard shortcuts
✅ Maintained by Textualize

### prompt_toolkit
⚠️ Lower-level
⚠️ More code to write
✅ Flexible
✅ Good for simple prompts

### Rich Console
⚠️ No async input
⚠️ Blocking
✅ Good for display only

### Typer (Current)
❌ Synchronous
❌ No real-time interaction
✅ Good for simple CLIs

## Next Steps

1. **Install Textual**: `pip install textual`
2. **Try the TUI**: `codur tui`
3. **Test commands**: Try `:pause`, `:resume`, `:hint`
4. **Integrate LangGraph**: Connect to real agent execution
5. **Add streaming**: Stream agent events to log pane

## Development

### Testing TUI Layout

```bash
# Run with development mode
textual run --dev codur/tui.py

# Opens browser with dev tools
# Shows CSS tree, event log, etc.
```

### Debugging

Textual has excellent debugging tools:

```bash
# Console for debugging
textual console

# In another terminal
python -m codur.tui_standalone

# Logs appear in console
```

### Customization

The TUI is styled with CSS-like syntax in `tui.py`:

```python
CSS = """
#log-container {
    height: 70%;
    border: solid $primary;
}
"""
```

Modify heights, colors, borders easily!

## Resources

- [Textual Documentation](https://textual.textualize.io/)
- [Textual Tutorial](https://textual.textualize.io/tutorial/)
- [Example Apps](https://github.com/Textualize/textual/tree/main/examples)
- [Textual Discord](https://discord.gg/Enf6Z3qhVr)

---

**The TUI is the future of Codur interaction!** 🚀
