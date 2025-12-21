#!/usr/bin/env python3
"""
Demo visualization of the Codur TUI

Shows what the TUI looks like and how it works
"""

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.table import Table
import time

console = Console()

PALETTE = {
    "primary": "#5cc8ff",
    "secondary": "#60a5fa",
    "accent": "#34d399",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "success": "#22c55e",
    "foreground": "#e2e8f0",
    "background": "#0b1020",
    "panel": "#0f172a",
}


def create_demo_layout():
    """Create the demo TUI layout"""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )

    layout["main"].split_row(
        Layout(name="log", ratio=2),
        Layout(name="help", ratio=1)
    )

    return layout


def render_header():
    """Render the header"""
    return Panel(
        f"[bold {PALETTE['primary']}]Codur Agent TUI[/bold {PALETTE['primary']}] - Autonomous Coding Agent",
        style=f"bold {PALETTE['foreground']} on {PALETTE['panel']}",
        border_style=PALETTE["primary"],
    )


def render_status(agent="Ollama", step="Planning", iteration=3):
    """Render the status bar"""
    status = Text()
    status.append("Agent: ", style=f"bold {PALETTE['primary']}")
    status.append(f"{agent} ", style=PALETTE["success"])
    status.append("│ Step: ", style=f"bold {PALETTE['primary']}")
    status.append(f"{step} ", style=PALETTE["warning"])
    status.append("│ Iteration: ", style=f"bold {PALETTE['primary']}")
    status.append(f"{iteration}", style=PALETTE["secondary"])

    return Panel(status, style=f"{PALETTE['foreground']} on {PALETTE['background']}", border_style=PALETTE["secondary"])


def render_log(messages):
    """Render the log pane"""
    log_text = "\n".join(messages)
    return Panel(
        log_text,
        title=f"[bold {PALETTE['primary']}]Agent Output[/bold {PALETTE['primary']}]",
        border_style=PALETTE["primary"],
        height=20
    )


def render_help():
    """Render the help/commands pane"""
    table = Table(title="Commands", show_header=False, box=None)
    table.add_column(style=PALETTE["primary"])
    table.add_column()

    table.add_row(":pause", "Pause agent")
    table.add_row(":resume", "Resume")
    table.add_row(":hint", "Add guidance")
    table.add_row(":set", "Switch agent")
    table.add_row("Ctrl+C", "Quit")

    return Panel(
        table,
        title=f"[bold {PALETTE['warning']}]Help[/bold {PALETTE['warning']}]",
        border_style=PALETTE["warning"],
    )


def render_input(text=""):
    """Render the input field"""
    return Panel(
        f"[bold {PALETTE['primary']}]>[/bold {PALETTE['primary']}] {text}█",
        title=f"[bold {PALETTE['accent']}]Input[/bold {PALETTE['accent']}]",
        border_style=PALETTE["accent"],
    )


def demo():
    """Run the demo"""
    print("\n" * 2)
    console.print("[bold cyan]Codur TUI Demo[/bold cyan]", justify="center")
    console.print("Showing what the interactive interface looks like\n", justify="center")

    time.sleep(2)

    layout = create_demo_layout()

    messages = [
        "[dim]16:30:45[/dim] [green]Codur Agent TUI started[/green]",
        "[dim]16:30:46[/dim] Loaded config with 4 agents",
        "",
    ]

    # Simulate a task execution
    steps = [
        ("Planning", "Analyzing task requirements..."),
        ("Planning", "Choosing optimal agent..."),
        ("Delegating", "Routing to Ollama (free local LLM)"),
        ("Executing", "Ollama generating code..."),
        ("Executing", "Generated 234 characters"),
        ("Reviewing", "Checking output quality..."),
        ("Complete", "Task completed successfully!"),
    ]

    user_inputs = [
        "",
        "",
        ":hint use recursion",
        "",
        "",
        ":pause",
        ":resume",
    ]

    with Live(layout, console=console, screen=True, refresh_per_second=4) as live:
        for i, (step, message) in enumerate(steps):
            # Update status
            layout["header"].update(render_header())
            layout["status"].update(render_status("Ollama", step, i + 1))

            # Add message to log
            timestamp = f"16:30:{45 + i:02d}"
            messages.append(f"[dim]{timestamp}[/dim] [blue]Step {i+1}:[/blue] {message}")

            # Show user input
            user_input = user_inputs[min(i, len(user_inputs) - 1)]
            if user_input:
                messages.append(f"[dim]{timestamp}[/dim] [cyan]User:[/cyan] {user_input}")

            # Update layout
            layout["main"]["log"].update(render_log(messages[-15:]))  # Last 15 messages
            layout["main"]["help"].update(render_help())
            layout["footer"].update(render_input(user_input))

            time.sleep(1.5)

    console.print("\n[bold green]Demo complete![/bold green]")
    console.print("\nTo try the real TUI:")
    console.print("  [cyan]codur tui[/cyan]\n")


if __name__ == "__main__":
    demo()
