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
    background: $panel;
}

#left-pane {
    width: 60%;
    layout: vertical;
    background: $panel;
}

#debug-pane {
    width: 40%;
    background: $panel;
    border: solid $secondary;
}

#debug-pane:focus-within {
    border: solid $accent;
    background: $panel;
}

#debug-pane.hidden {
    display: none;
}

#log-container {
    height: 70%;
    background: $panel;
    border: solid $primary;
}

#log-container:focus-within {
    border: solid $accent;
    background: $panel;
}

#input-container {
    height: 30%;
    min-height: 5;
    background: $panel;
    border: solid $primary;
}

#input-container:focus-within {
    border: solid $accent;
}

RichLog {
    border: none;
    background: $panel;
}

#log,
#debug-log {
    background: $panel;
}

#log:focus,
#debug-log:focus {
    background: $panel;
}

TextArea {
    border: none;
    background: $panel;
    height: 3;
    min-height: 3;
}

#log:focus,
#debug-log:focus,
RichLog:focus,
RichLog:focus-within,
TextArea:focus {
    background: $panel;
}

#controls {
    height: 1;
    dock: bottom;
    padding: 0 1;
    background: $panel;
    color: $foreground;
}

Screen.fullscreen Header {
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

Screen.fullscreen #controls {
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
