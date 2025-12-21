"""Shared TUI widgets and screens."""

from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static, Input, ListView, ListItem, Label


class AgentStatus(Static):
    """Widget showing current agent status."""

    def __init__(self):
        super().__init__()
        self.current_agent = "None"
        self.current_step = "Idle"
        self.iterations = 0

    def update_status(self, agent: str, step: str, iterations: int):
        """Update the status display."""
        self.current_agent = agent
        self.current_step = step
        self.iterations = iterations
        self.update(self.render_status())

    def render_status(self) -> Text:
        """Render the status text."""
        text = Text()
        text.append("Agent: ", style="bold cyan")
        text.append(f"{self.current_agent} ", style="green")
        text.append("| Step: ", style="bold cyan")
        text.append(f"{self.current_step} ", style="yellow")
        text.append("| Iteration: ", style="bold cyan")
        text.append(f"{self.iterations}", style="magenta")
        return text


class FileSearchScreen(ModalScreen[Optional[str]]):
    """Modal screen to fuzzy-search and insert file paths."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "select", "Select"),
    ]

    def __init__(self, files: list[str]):
        super().__init__()
        self._files = files
        self._matches: list[str] = []

    def compose(self) -> ComposeResult:
        with Container(id="file-search-dialog"):
            yield Static("Search files & directories", id="file-search-title")
            yield Input(placeholder="Type to filter...", id="file-search-input")
            yield ListView(id="file-search-list")
            yield Static("📁 = directory | Enter to insert, Esc to cancel", id="file-search-help")

    def on_mount(self) -> None:
        self._update_matches("")
        self.query_one("#file-search-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "file-search-input":
            return
        self._update_matches(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "file-search-input":
            return
        event.stop()
        if self._matches:
            self.dismiss(self._matches[0])
        else:
            self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_view = event.list_view
        index = list_view.index
        if index is None:
            return
        if index >= len(self._matches):
            return
        self.dismiss(self._matches[index])

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        list_view = self.query_one("#file-search-list", ListView)
        index = list_view.index
        if index is None:
            if self._matches:
                self.dismiss(self._matches[0])
            return
        if index >= len(self._matches):
            return
        self.dismiss(self._matches[index])

    def _update_matches(self, query: str) -> None:
        list_view = self.query_one("#file-search-list", ListView)
        matches = self._filter_matches(query)
        self._matches = matches
        list_view.clear()
        for match in matches[:200]:
            # Add visual indicator for directories
            display_text = f"📁 {match}" if match.endswith("/") else match
            list_view.append(ListItem(Label(display_text)))
        list_view.index = 0 if matches else None

    def _filter_matches(self, query: str) -> list[str]:
        needle = query.strip().lower()
        if not needle:
            return sorted(self._files)[:200]
        scored: list[tuple[float, str]] = []
        for path in self._files:
            score = self._fuzzy_score(needle, path.lower())
            if score is None:
                continue
            scored.append((score, path))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:200]]

    def _fuzzy_score(self, needle: str, haystack: str) -> Optional[float]:
        pos = -1
        score = 0.0
        for char in needle:
            pos = haystack.find(char, pos + 1)
            if pos == -1:
                return None
            score += 2.0
            if pos == 0 or haystack[pos - 1] in ("/", "_", "-", "."):
                score += 1.0
        score -= len(haystack) * 0.01
        if needle in haystack:
            score += 5.0
        return score
