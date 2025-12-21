#!/usr/bin/env python3
"""
Codur CLI - Command-line interface for the coding agent
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
    module="langchain_core._api.deprecation",
)

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from datetime import datetime
from typing import Optional
from pathlib import Path

from codur.graph.main_graph import create_agent_graph
from codur.config import load_config, save_config
from langchain_core.messages import HumanMessage
from codur.model_registry import (
    list_groq_models,
    list_openai_models,
    list_anthropic_models,
    list_ollama_models,
    list_ollama_registry_models,
)

app = typer.Typer(
    name="codur",
    help="Autonomous coding agent orchestrator",
    add_completion=False,
)
console = Console()

def _run_prompt(prompt: str, config: Optional[Path], verbose: bool, raw: bool) -> None:
    if not raw:
        console.print(Panel.fit(
            "[bold cyan]Codur Agent[/bold cyan]\n"
            f"Task: {prompt}",
            border_style="cyan"
        ))

    cfg = load_config(config)
    graph = create_agent_graph(cfg)

    try:
        result = graph.invoke({
            "messages": [HumanMessage(content=prompt)],
            "verbose": verbose,
            "config": cfg,
        })

        selected_agent = result.get("selected_agent")
        if raw:
            if selected_agent:
                console.print(f"[dim]Selected agent:[/dim] {selected_agent}")
            console.print(result.get("final_response", "No response generated"))
        else:
            if selected_agent:
                console.print(f"[dim]Selected agent:[/dim] {selected_agent}")
            console.print("\n[bold green]Result:[/bold green]")
            console.print(result.get("final_response", "No response generated"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}", style="red")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    command: Optional[str] = typer.Option(
        None,
        "-c",
        "--command",
        help="Run a single prompt without subcommands",
    ),
    raw: bool = typer.Option(
        False,
        "-r",
        "--raw",
        help="Only print the response content",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to config file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
):
    if command:
        _run_prompt(command, config, verbose, raw)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task to execute"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    raw: bool = typer.Option(
        False,
        "-r",
        "--raw",
        help="Only print the response content",
    ),
):
    """
    Run a coding task through the agent orchestrator.

    Example:
        codur run "Create a Python function to calculate fibonacci numbers"
        codur run "Refactor the authentication module" --verbose
    """
    _run_prompt(prompt, config, verbose, raw)


@app.command()
def configure(
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider to use for planning (groq, openai, anthropic, ollama)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model name for planning",
    ),
    list_models: bool = typer.Option(
        False,
        "--list-models",
        help="List models for the chosen provider",
    ),
    list_model_registry: bool = typer.Option(
        False,
        "--list-model-registry",
        help="List models from the Ollama registry (provider=ollama only)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file to update",
    ),
):
    """
    Configure the planning LLM provider and model.
    """
    cfg_path = config or Path("codur.yaml")
    cfg = load_config(cfg_path if cfg_path.exists() else None)

    if not provider:
        provider = Prompt.ask(
            "Choose provider",
            choices=["groq", "openai", "anthropic", "ollama"],
            default="groq",
        )

    provider = provider.lower()

    if list_models or list_model_registry:
        if provider == "groq":
            models = list_groq_models()
            console.print("\n".join(models))
        elif provider == "openai":
            models = list_openai_models()
            for item in models:
                created = item.get("created", 0)
                created_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d") if created else "unknown"
                console.print(f"{item.get('id')} ({created_str})")
        elif provider == "anthropic":
            models = list_anthropic_models()
            console.print("\n".join(models))
        elif provider == "ollama":
            if list_model_registry:
                max_size = None
                ollama_provider = cfg.providers.get("ollama")
                if ollama_provider:
                    max_size = ollama_provider.max_model_size_gb
                models = list_ollama_registry_models(max_size_gb=max_size)
            else:
                base_url = None
                ollama_provider = cfg.providers.get("ollama")
                if ollama_provider and ollama_provider.base_url:
                    base_url = ollama_provider.base_url
                models = list_ollama_models(base_url=base_url)
            console.print("\n".join(models))
        else:
            console.print(f"No model listing available for provider: {provider}")
        raise typer.Exit()

    if not model:
        default_model = cfg.llm.profiles.get(cfg.llm.default_profile or "", None)
        model = Prompt.ask(
            "Model name",
            default=default_model.model if default_model else "",
        )

    profile_name = f"{provider}-{model}"
    existing_profile = cfg.llm.profiles.get(profile_name)
    cfg.llm.profiles[profile_name] = {
        "provider": provider,
        "model": model,
        "temperature": existing_profile.temperature if existing_profile else None,
    }
    cfg.llm.default_profile = profile_name

    if cfg_path.exists():
        save_config(cfg, cfg_path)
        console.print(f"Updated config at {cfg_path}")
    else:
        console.print("No codur.yaml found; writing to ~/.codur/config.yaml")
        save_config(cfg, None)


@app.command()
def list_agents():
    """List available agents and their capabilities."""
    console.print("[bold cyan]Available Agents:[/bold cyan]\n")

    agents = [
        ("ollama", "Local LLM for code generation (FREE)", "Simple code generation, explanations"),
        ("groq", "Groq hosted LLM", "Fast code generation and reasoning"),
        ("claude_code", "Claude Code CLI integration", "Multi-file changes, complex reasoning, tool usage"),
        ("codex", "OpenAI Codex for refactoring", "Code refactoring, bug fixes, optimization"),
        ("sheets", "Google Sheets integration", "Read/write spreadsheet data"),
        ("linkedin", "LinkedIn job scraper", "Job search and scraping"),
    ]

    for name, desc, capabilities in agents:
        console.print(f"[green]●[/green] [bold]{name}[/bold]")
        console.print(f"  {desc}")
        console.print(f"  [dim]Capabilities: {capabilities}[/dim]\n")


@app.command()
def list_mcp():
    """List configured MCP servers."""
    console.print("[bold cyan]Configured MCP Servers:[/bold cyan]\n")

    cfg = load_config()

    if not cfg.mcp_servers:
        console.print("[yellow]No MCP servers configured[/yellow]")
        return

    for name, server_cfg in cfg.mcp_servers.items():
        console.print(f"[green]●[/green] [bold]{name}[/bold]")
        console.print(f"  Command: {server_cfg.get('command', 'N/A')}")
        console.print(f"  Status: [dim]Available[/dim]\n")


@app.command()
def interactive():
    """
    Start an interactive session with the coding agent.
    """
    console.print(Panel.fit(
        "[bold cyan]Codur Interactive Mode[/bold cyan]\n"
        "Type your tasks, or 'quit' to exit",
        border_style="cyan"
    ))

    cfg = load_config()
    graph = create_agent_graph(cfg)

    while True:
        try:
            prompt = console.input("\n[bold cyan]codur>[/bold cyan] ")

            if prompt.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not prompt.strip():
                continue

            result = graph.invoke({
                "messages": [HumanMessage(content=prompt)],
                "config": cfg,
            })

            selected_agent = result.get("selected_agent")
            if selected_agent:
                console.print(f"[dim]Selected agent:[/dim] {selected_agent}")
            console.print(f"\n[green]{result.get('final_response', 'No response')}[/green]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def tui(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
):
    """
    Launch the interactive TUI (Terminal User Interface).

    The TUI provides:
    - Real-time agent progress display
    - Live log streaming
    - Interactive command input
    - Ability to pause/resume agents
    - Mid-execution guidance
    """
    from codur.tui import run_tui
    from codur.config import load_config

    cfg = load_config(config)
    run_tui(cfg)


@app.command()
def version():
    """Show version information."""
    from codur import __version__
    console.print(f"[bold]Codur[/bold] version [cyan]{__version__}[/cyan]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
