"""Textual CSS for the Codur TUI."""

TUI_CSS = """
Screen {
    layout: vertical;
    background: $background;
    color: $foreground;
}

#status {
    dock: top;
    height: 3;
    background: $panel;
    padding: 1;
}

#main {
    layout: horizontal;
}

#left-pane {
    width: 60%;
    layout: vertical;
}

#debug-pane {
    width: 40%;
    border: solid $secondary;
}

#debug-pane.hidden {
    display: none;
}

#log-container {
    height: 70%;
    border: solid $primary;
}

#input-container {
    height: 30%;
    border: solid $accent;
}

RichLog {
    border: none;
}

Input {
    border: none;
}

#controls {
    height: 3;
    layout: horizontal;
}

Screen.fullscreen Header {
    display: none;
}

Screen.fullscreen Footer {
    display: none;
}

Screen.fullscreen #status {
    display: none;
}

Screen.fullscreen #debug-pane {
    display: none;
}

Screen.fullscreen #input-container {
    display: none;
}

Screen.fullscreen #main {
    layout: vertical;
}

Screen.fullscreen #left-pane {
    width: 100%;
}

Screen.fullscreen #log-container {
    height: 100%;
    border: none;
}

Screen.fullscreen RichLog {
    border: none;
}

FileSearchScreen {
    align: center middle;
}

#file-search-dialog {
    width: 70%;
    height: 70%;
    background: $panel;
    border: round $primary;
    padding: 1 2;
}

#file-search-list {
    height: 1fr;
    border: solid $primary;
}

#file-search-help {
    height: 1;
    color: $text-muted;
}
"""
