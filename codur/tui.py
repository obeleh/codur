"""
Textual TUI for Codur - Async interactive interface

Provides a split-pane interface where users can:
- See agent progress in real-time (top pane)
- Enter commands and guidance (bottom pane)
- Interrupt and guide agents mid-execution
"""

import asyncio
import warnings
import threading
import os
import re
from datetime import datetime
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
    module="langchain_core._api.deprecation",
)

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, TextArea, RichLog, Static
from textual import events
from textual.binding import Binding
from textual.theme import Theme
from rich.panel import Panel
from typing import Optional

from codur.tools.filesystem import EXCLUDE_DIRS
from codur.tui_components import AgentStatus, FileSearchScreen
from codur.tui_style import TUI_CSS

from codur.config import load_config, CodurConfig
from codur.graph.main_graph import create_agent_graph
from langchain_core.messages import HumanMessage


class CommandInput(TextArea):
    BINDINGS = [
        ("ctrl+enter", "submit", "Submit"),
        ("ctrl+shift+m", "submit", "Submit"),
    ]

    async def action_submit(self) -> None:
        await self.app.action_submit()

    async def on_key(self, event: events.Key) -> None:
        aliases = getattr(event, "aliases", []) or []
        is_ctrl_enter = (
            event.key in {"ctrl+enter", "ctrl+shift+m", "ctrl+m"}
            or (event.key == "enter" and any(alias.startswith("ctrl") for alias in aliases))
        )
        if is_ctrl_enter:
            event.stop()
            await self.app.action_submit()


class CodurTUI(App):
    """
    Textual TUI for Codur agent interaction.

    Features:
    - Top pane: Real-time agent logs and progress
    - Bottom pane: User input for commands and guidance
    - Status bar: Current agent and step
    """

    CSS = TUI_CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+p", "pause", "Pause", show=True),
        Binding("ctrl+r", "resume", "Resume", show=True),
        Binding("ctrl+d", "toggle_debug", "Debug", show=True),
        Binding("ctrl+f", "toggle_fullscreen", "Fullscreen", show=True),
        Binding("ctrl+t", "toggle_timestamps", "Timestamps", show=True),
        Binding("ctrl+a", "select_all", "Select All", show=True),
        Binding("ctrl+enter", "submit", "Submit", show=True),
        ("escape", "clear_input", "Clear"),
    ]

    def __init__(self, config: Optional[CodurConfig] = None):
        super().__init__()
        self.config = config or load_config()
        self._codur_theme = Theme(
            name="codur",
            primary="#5cc8ff",
            secondary="#60a5fa",
            accent="#34d399",
            warning="#f59e0b",
            error="#ef4444",
            success="#22c55e",
            foreground="#e2e8f0",
            background="#0b1020",
            surface="#111827",
            panel="#0f172a",
            dark=True,
        )
        self.graph = create_agent_graph(self.config)
        self.agent_tasks: list[asyncio.Task] = []
        self.paused = False
        self.debug_visible = True  # Show debug panel by default
        self.fullscreen = False
        self.show_timestamps = True
        self.log_history: list[dict] = []
        self.debug_history: list[dict] = []
        self.user_queue: asyncio.Queue = asyncio.Queue()
        self._file_index: list[str] = []
        self._file_search_active = False
        self._last_input_value = ""
        self._file_insert_pos: Optional[int] = None

    def _is_quick_response(self, task: str) -> bool:
        """Detect small-talk or trivial input to avoid noisy step logs."""
        text = task.strip().lower()
        if not text:
            return True
        greetings = {
            "hi", "hello", "hey", "yo", "sup",
            "good morning", "good afternoon", "good evening",
            "thanks", "thank you",
        }
        if text in greetings:
            return True
        if len(text.split()) <= 3 and any(word in greetings for word in text.split()):
            return True
        return False

    def compose(self) -> ComposeResult:
        """Compose the UI layout"""
        yield Header()

        # Status bar
        yield Container(
            AgentStatus(),
            id="status"
        )

        # Main content - horizontal layout with main pane and debug pane
        with Container(id="main"):
            # Left pane: logs and input
            with Container(id="left-pane"):
                # Log output (top 70%)
                with Container(id="log-container"):
                    yield RichLog(
                        id="log",
                        highlight=True,
                        markup=True,
                        wrap=True,
                    )

                # User input (bottom 30%)
                with Container(id="input-container"):
                    yield CommandInput(
                        "",
                        placeholder="Enter task or command... (use @ to insert a file)",
                        id="input"
                    )

            # Right pane: debug panel (LLM communications)
            with Container(id="debug-pane"):
                yield Static("[bold yellow]🔍 LLM Communications Debug Panel[/bold yellow]\n", id="debug-header")
                yield RichLog(
                    id="debug-log",
                    highlight=True,
                    markup=True,
                    wrap=True,
                )

        yield Static(
            "[bold cyan]Commands:[/] :pause :resume :hint <text> :set agent=<name> | @ file insert | Ctrl+Enter Submit | Ctrl+D Debug | Ctrl+F Fullscreen",
            id="controls",
        )

    def on_mount(self) -> None:
        """Called when app starts"""
        self.register_theme(self._codur_theme)
        self.theme = "codur"
        self.log_message("[bold green]Codur Agent TUI[/bold green]")
        self.log_message("Type a task to start, or use commands like :pause, :resume")
        enabled_configs = [name for name, cfg in self.config.agents.configs.items() if cfg.enabled]
        enabled_profiles = [name for name, cfg in self.config.agents.profiles.items() if cfg.enabled]
        total_enabled = len(enabled_configs) + len(enabled_profiles)
        self.log_message(f"Loaded config with {total_enabled} agents\n")

        # Focus the input
        self.query_one("#input", TextArea).focus()
        asyncio.create_task(self._build_file_index())

    async def _build_file_index(self) -> None:
        self._file_index = await asyncio.to_thread(self._scan_files)

    def _load_gitignore(self, root: str) -> PathSpec | None:
        """Load and parse .gitignore patterns from the repository root."""
        gitignore_path = os.path.join(root, ".gitignore")
        if not os.path.exists(gitignore_path):
            return None

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                patterns = f.read().splitlines()
            # Filter out empty lines and comments
            patterns = [p for p in patterns if p.strip() and not p.strip().startswith("#")]
            return PathSpec.from_lines(GitWildMatchPattern, patterns)
        except (IOError, OSError):
            return None

    def _is_ignored(self, path: str, spec: PathSpec | None, is_dir: bool = False) -> bool:
        """Check if a path is ignored by gitignore patterns or hardcoded excludes."""
        # Check hardcoded exclude directories
        if is_dir and os.path.basename(path) in EXCLUDE_DIRS:
            return True

        # Check gitignore patterns
        if spec:
            # For directories, check both with and without trailing slash
            if is_dir:
                return spec.match_file(path) or spec.match_file(path + "/")
            return spec.match_file(path)

        return False

    def _scan_files(self) -> list[str]:
        root = os.getcwd()
        results: list[str] = []

        # Load .gitignore patterns
        gitignore_spec = self._load_gitignore(root)

        for dirpath, dirnames, filenames in os.walk(root):
            # Filter directories to exclude both hardcoded and gitignored ones
            dirnames[:] = [
                d for d in dirnames
                if not self._is_ignored(
                    os.path.relpath(os.path.join(dirpath, d), root),
                    gitignore_spec,
                    is_dir=True
                )
            ]

            # Add directories
            for dirname in dirnames:
                full_path = os.path.join(dirpath, dirname)
                rel_path = os.path.relpath(full_path, root)
                results.append(rel_path + "/")  # Append / to indicate directory

            # Add files (excluding ignored ones)
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root)
                if not self._is_ignored(rel_path, gitignore_spec, is_dir=False):
                    results.append(rel_path)

        return results

    def _annotate_file_mentions(self, task: str) -> str:
        if not re.search(r"@\\S", task):
            return task
        note = "Note: tokens starting with @ are file paths."
        if note in task:
            return task
        return f"{task}\\n\\n{note}"

    @staticmethod
    def _cursor_index(text: str, location: tuple[int, int]) -> int:
        row, column = location
        lines = text.splitlines(keepends=True)
        if row <= 0:
            return min(column, len(text))
        total = 0
        for idx, line in enumerate(lines):
            if idx == row:
                return min(total + column, len(text))
            total += len(line)
        return len(text)

    @staticmethod
    def _cursor_location_from_index(text: str, index: int) -> tuple[int, int]:
        if index < 0:
            index = 0
        if index > len(text):
            index = len(text)
        lines = text.splitlines(keepends=True)
        if not lines:
            return (0, 0)
        total = 0
        for row, line in enumerate(lines):
            next_total = total + len(line)
            if index <= next_total:
                return (row, index - total)
            total = next_total
        return (len(lines) - 1, len(lines[-1]))

    def log_message(self, message: str, style: str = ""):
        """Add a message to the log"""
        log = self.query_one("#log", RichLog)
        message = message.strip("\n")
        if not message:
            return

        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "style": style,
        }
        self.log_history.append(entry)
        if len(self.log_history) > 1000:
            self.log_history.pop(0)
        self._write_log_entry(log, entry, self.show_timestamps)

    def log_debug(self, message: str, style: str = ""):
        """Add a message to the debug log"""
        try:
            debug_log = self.query_one("#debug-log", RichLog)
            message = message.strip("\n")
            if not message:
                return

            entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": message,
                "style": style,
            }
            self.debug_history.append(entry)
            if len(self.debug_history) > 1000:
                self.debug_history.pop(0)
            self._write_log_entry(debug_log, entry, self.show_timestamps)
        except Exception:
            pass  # Debug panel might not be visible

    def _write_log_entry(self, log: RichLog, entry: dict, show_timestamps: bool) -> None:
        timestamp = entry["timestamp"]
        message = entry["message"]
        style = entry["style"]
        if style:
            if show_timestamps:
                log.write(f"[dim]{timestamp}[/dim] [{style}]{message}[/{style}]")
            else:
                log.write(f"[{style}]{message}[/{style}]")
        else:
            if show_timestamps:
                log.write(f"[dim]{timestamp}[/dim] {message}")
            else:
                log.write(message)

    def _render_logs(self) -> None:
        try:
            log = self.query_one("#log", RichLog)
        except Exception:
            return
        log.clear()
        for entry in self.log_history:
            self._write_log_entry(log, entry, self.show_timestamps)
        try:
            debug_log = self.query_one("#debug-log", RichLog)
        except Exception:
            return
        debug_log.clear()
        for entry in self.debug_history:
            self._write_log_entry(debug_log, entry, self.show_timestamps)

    def update_agent_status(self, agent: str, step: str, iterations: int):
        """Update the status bar"""
        status = self.query_one(AgentStatus)
        status.update_status(agent, step, iterations)

    async def action_submit(self) -> None:
        """Handle user input submission."""
        input_widget = self.query_one("#input", TextArea)
        user_input = input_widget.text.strip()

        if not user_input:
            return

        input_widget.text = ""
        input_widget.cursor_location = (0, 0)
        self._last_input_value = ""

        if user_input.startswith(":"):
            await self.handle_command(user_input)
        else:
            await self.start_task(user_input)
        # Keep focus on the input after submit
        input_widget.focus()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "input":
            return
        value = event.text_area.text
        if not self._file_search_active:
            if len(value) - len(self._last_input_value) > 1:
                trimmed_lines = "\n".join(line.rstrip() for line in value.splitlines())
                if trimmed_lines != value:
                    event.text_area.text = trimmed_lines
                    event.text_area.cursor_location = self._cursor_location_from_index(
                        trimmed_lines,
                        len(trimmed_lines),
                    )
                    self._last_input_value = trimmed_lines
                    return
            trailing = re.search(r"\s+$", value)
            if trailing and len(trailing.group(0)) > 3:
                trimmed = value[:trailing.start()] + " "
                event.text_area.text = trimmed
                event.text_area.cursor_location = self._cursor_location_from_index(
                    trimmed,
                    len(trimmed),
                )
                self._last_input_value = trimmed
                return
        if self._file_search_active:
            self._last_input_value = value
            return
        cursor_pos = self._cursor_index(value, event.text_area.cursor_location)
        inserted_at = cursor_pos - 1 if cursor_pos > 0 else None
        is_new_char = len(value) == len(self._last_input_value) + 1
        typed_at = inserted_at is not None and inserted_at < len(value)
        inserted_char = value[inserted_at] if typed_at else None
        if is_new_char and inserted_char == "@":
            self._file_search_active = True
            self._file_insert_pos = inserted_at
            self.push_screen(
                FileSearchScreen(self._file_index),
                callback=self._on_file_selected,
            )
        self._last_input_value = event.text_area.text

    async def on_key(self, event) -> None:
        if event.key not in {"ctrl+enter", "ctrl+shift+m"}:
            return
        focused = self.focused
        if focused and getattr(focused, "id", None) == "input":
            event.stop()
            await self.action_submit()

    def _on_file_selected(self, selection: Optional[str]) -> None:
        input_widget = self.query_one("#input", TextArea)
        if selection:
            value = input_widget.text
            insert_pos = self._file_insert_pos
            if insert_pos is None or insert_pos > len(value):
                insert_pos = len(value)
            if insert_pos < len(value) and value[insert_pos] == "@":
                prefix = value[:insert_pos]
                suffix = value[insert_pos + 1:]
            else:
                prefix = value[:insert_pos]
                suffix = value[insert_pos:]
            spacer = "" if (suffix and suffix[0].isspace()) else " "
            updated = f"{prefix}@{selection}{spacer}{suffix}"
            input_widget.text = updated
            input_widget.cursor_location = self._cursor_location_from_index(
                updated,
                len(prefix) + len(selection) + 1 + len(spacer),
            )
        self._file_insert_pos = None
        self._file_search_active = False

    async def handle_command(self, command: str):
        """Handle special commands"""
        cmd = command.lower()

        if cmd == ":pause":
            self.paused = True
            self.log_message("[yellow]Agent paused. Type :resume to continue.[/yellow]")

        elif cmd == ":resume":
            self.paused = False
            self.log_message("[green]Agent resumed.[/green]")

        elif cmd.startswith(":hint "):
            hint = command[6:].strip()
            await self.user_queue.put({"type": "hint", "content": hint})
            self.log_message(f"[cyan]Hint added:[/cyan] {hint}")

        elif cmd.startswith(":set agent="):
            agent = command[11:].strip()
            await self.user_queue.put({"type": "set_agent", "agent": agent})
            self.log_message(f"[cyan]Agent set to:[/cyan] {agent}")

        elif cmd == ":help":
            self.show_help()

        else:
            self.log_message(f"[red]Unknown command:[/red] {command}")

    def show_help(self):
        """Show help message"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]
  :pause          - Pause agent execution
  :resume         - Resume agent execution
  :hint <text>    - Provide guidance to agent
  :set agent=<>   - Switch to specific agent
  @              - Fuzzy search and insert file paths
  :help           - Show this help
        """
        self.log_message(help_text)

    async def start_task(self, task: str):
        """Start a new agent task"""
        self.log_message(f"\n[bold green]Starting task:[/bold green] {task}\n")

        # Create and start the agent task
        annotated_task = self._annotate_file_mentions(task)
        new_task = asyncio.create_task(self.run_agent(annotated_task))
        self.agent_tasks.append(new_task)
        new_task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        """Cleanup when a background task finishes."""
        try:
            self.agent_tasks.remove(task)
        except ValueError:
            pass

    async def run_agent(self, task: str):
        """Run the agent in background"""
        try:
            if self._is_quick_response(task):
                await self._run_quick_response(task)
            else:
                await self._run_streaming_task(task)

            self.log_message("\n[bold green]Task completed![/bold green]\n")
            self.update_agent_status("None", "Idle", 0)

        except Exception as e:
            import traceback
            self.log_message(f"[bold red]Error:[/bold red] {str(e)}")
            self.log_debug(f"[bold red]Error:[/bold red] {str(e)}\n{traceback.format_exc()}", "")
            self.update_agent_status("Error", "Failed", 0)

    async def _run_quick_response(self, task: str) -> None:
        self.update_agent_status("Orchestrator", "Responding", 0)
        result = await asyncio.to_thread(
            self.graph.invoke,
            {"messages": [HumanMessage(content=task)], "config": self.config},
        )
        self.log_message("\n[bold green]Response:[/bold green]")
        self.log_message(result.get("final_response", "No response generated"))

    async def _run_streaming_task(self, task: str) -> None:
        self.update_agent_status("Starting", "Planning", 0)

        initial_state = {
            "messages": [HumanMessage(content=task)],
            "iterations": 0,
            "verbose": self.config.verbose,
            "config": self.config,
        }

        self.log_debug("[bold cyan]📨 User Message:[/bold cyan]", "")
        self.log_debug(f"  {task}\n", "green")
        planner_details = self._describe_planner_llm()
        if planner_details:
            self.log_debug(f"  planner_llm: {planner_details}", "yellow")

        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        thread = threading.Thread(
            target=self._stream_worker,
            args=(initial_state, event_queue, loop),
            daemon=True,
        )
        thread.start()

        step_num = 0
        while True:
            event = await event_queue.get()
            if event is None:
                break

            if self.paused:
                self.log_message("[yellow]Waiting for resume...[/yellow]")
                while self.paused:
                    await asyncio.sleep(0.1)

            step_num += 1
            self._handle_stream_event(event, step_num)
            await asyncio.sleep(0.05)

    def _stream_worker(self, initial_state: dict, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        try:
            for event in self.graph.stream(initial_state):
                loop.call_soon_threadsafe(event_queue.put_nowait, event)
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    def _handle_stream_event(self, event: dict, step_num: int) -> None:
        self.log_debug(f"\n[bold yellow]📊 Step {step_num} Event:[/bold yellow]", "")
        for node_name, node_output in event.items():
            self.log_debug(f"  Node: [bold]{node_name}[/bold]", "cyan")
            self._log_node_output(node_name, node_output, step_num)

        if event:
            node_name = list(event.keys())[0]
            self.update_agent_status("Agent", node_name.title(), step_num)

    def _log_node_output(self, node_name: str, node_output: dict, step_num: int) -> None:
        if "llm_debug" in node_output:
            llm_debug = node_output["llm_debug"]
            self.log_debug(f"\n  [bold magenta]🤖 LLM Communication:[/bold magenta]", "")

            # Display LLM model and profile info
            llm_model = llm_debug.get('llm_model', 'N/A')
            llm_profile = llm_debug.get('llm_profile', 'N/A')
            self.log_debug(f"  [bold]Model:[/bold] {llm_model} [dim](profile: {llm_profile})[/dim]", "blue")

            # Display errors prominently if present
            if "error" in llm_debug:
                error = llm_debug["error"]
                self.log_debug(f"\n  [bold red on yellow]⚠️  PARSING ERROR ⚠️[/bold red on yellow]", "")
                self.log_debug(f"  [bold red]Type:[/bold red] {error.get('type', 'Unknown')}", "")
                self.log_debug(f"  [bold red]Message:[/bold red] {error.get('message', 'N/A')}", "")
                self.log_debug(f"  [bold red]Raw Response:[/bold red]", "")
                self.log_debug(f"    {error.get('raw_response', 'N/A')}\n", "yellow")

            self.log_debug(f"  [bold]System Prompt:[/bold]", "cyan")
            self.log_debug(f"    {llm_debug.get('system_prompt', 'N/A')}\n", "dim")
            self.log_debug(f"  [bold]User Message:[/bold]", "cyan")
            self.log_debug(f"    {llm_debug.get('user_message', 'N/A')}\n", "green")
            self.log_debug(f"  [bold]LLM Response:[/bold]", "cyan")
            self.log_debug(f"    {llm_debug.get('llm_response', 'N/A')}\n", "yellow")

        if "next_action" in node_output:
            action = node_output["next_action"]
            self.log_message(f"[blue]Next action:[/blue] {action}")
            self.log_debug(f"    next_action: {action}", "yellow")

        if "selected_agent" in node_output:
            agent = node_output["selected_agent"]
            self.log_message(f"[cyan]Selected agent:[/cyan] {agent}")
            self.log_debug(f"    selected_agent: {agent}", "green")
            agent_details = self._describe_agent(agent)
            if agent_details:
                self.log_debug(f"    agent_details: {agent_details}", "yellow")
            self.update_agent_status(agent, node_name, step_num)

        if "tool_calls" in node_output:
            tool_calls = node_output["tool_calls"]
            self.log_debug(f"    tool_calls: {tool_calls}", "cyan")

        if "final_response" in node_output:
            response = node_output["final_response"]
            self.log_message(f"\n[bold green]Response:[/bold green]\n{response}\n")
            self.log_debug(f"    final_response: {response}", "green")

        if "agent_outcome" in node_output:
            outcome = node_output["agent_outcome"]
            if "result" in outcome:
                result = outcome["result"]
                if len(result) > 500:
                    self.log_message(f"\n[green]Result:[/green]\n{result[:500]}...\n")
                else:
                    self.log_message(f"\n[green]Result:[/green]\n{result}\n")
                self.log_debug(f"    agent_outcome: {outcome}", "magenta")
            agent_name = outcome.get("agent")
            if agent_name:
                self.log_debug(f"    sent_to: {agent_name}", "yellow")
                agent_details = self._describe_agent(agent_name)
                if agent_details:
                    self.log_debug(f"    agent_details: {agent_details}", "yellow")

    def action_pause(self) -> None:
        """Pause agent execution"""
        self.paused = True
        self.log_message("[yellow]Agent paused via Ctrl+P[/yellow]")

    def action_resume(self) -> None:
        """Resume agent execution"""
        self.paused = False
        self.log_message("[green]Agent resumed via Ctrl+R[/green]")

    def action_clear_input(self) -> None:
        """Clear the input field"""
        self.query_one("#input", TextArea).text = ""

    def action_toggle_debug(self) -> None:
        """Toggle debug panel visibility"""
        debug_pane = self.query_one("#debug-pane")
        self.debug_visible = not self.debug_visible

        if self.debug_visible:
            debug_pane.remove_class("hidden")
            self.log_message("[yellow]Debug panel shown (Ctrl+D to hide)[/yellow]")
        else:
            debug_pane.add_class("hidden")
            self.log_message("[yellow]Debug panel hidden (Ctrl+D to show)[/yellow]")

    def action_toggle_fullscreen(self) -> None:
        """Toggle fullscreen log view."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen.add_class("fullscreen")
            self.log_message("[yellow]Fullscreen log view enabled (Ctrl+F to exit)[/yellow]")
        else:
            self.screen.remove_class("fullscreen")
            self.log_message("[yellow]Fullscreen log view disabled (Ctrl+F to enable)[/yellow]")

    def action_toggle_timestamps(self) -> None:
        """Toggle timestamps in logs."""
        self.show_timestamps = not self.show_timestamps
        self._render_logs()
        state = "enabled" if self.show_timestamps else "disabled"
        self.log_message(f"[yellow]Timestamps {state} (Ctrl+T to toggle)[/yellow]")

    def action_select_all(self) -> None:
        """Select all text in the input widget."""
        try:
            input_widget = self.query_one("#input", TextArea)
        except Exception:
            return
        input_widget.action_select_all()

    def _describe_agent(self, agent_name: str) -> str:
        if not agent_name:
            return ""
        if agent_name.startswith("llm:"):
            profile_name = agent_name.split(":", 1)[1]
            profile = self.config.llm.profiles.get(profile_name)
            if profile:
                return f"llm profile={profile_name} provider={profile.provider} model={profile.model}"
            return f"llm profile={profile_name}"
        if agent_name.startswith("agent:"):
            profile_name = agent_name.split(":", 1)[1]
            profile = self.config.agents.profiles.get(profile_name)
            if profile:
                model = profile.config.get("model")
                if model:
                    return f"agent profile={profile_name} name={profile.name} model={model}"
                return f"agent profile={profile_name} name={profile.name}"
            return f"agent profile={profile_name}"
        config = self.config.agents.configs.get(agent_name)
        if config:
            model = config.config.get("model")
            if model:
                return f"agent {agent_name} model={model}"
            return f"agent {agent_name}"
        return ""

    def _describe_planner_llm(self) -> str:
        profile_name = self.config.llm.default_profile
        if not profile_name:
            return ""
        profile = self.config.llm.profiles.get(profile_name)
        if profile:
            return f"profile={profile_name} provider={profile.provider} model={profile.model}"
        return f"profile={profile_name}"


def run_tui(config: Optional[CodurConfig] = None):
    """Run the Textual TUI"""
    app = CodurTUI(config=config)
    app.run()


if __name__ == "__main__":
    run_tui()
