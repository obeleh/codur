#!/usr/bin/env python3
"""
Codur CLI - Command-line interface for the coding agent
"""

import warnings
import os
from tabnanny import verbose

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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
console = Console(stderr=False)


def _invoke_graph(graph, payload: dict, timeout_s: int | None):
    if not timeout_s:
        return graph.invoke(payload)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(graph.invoke, payload)
    timed_out = False
    try:
        return future.result(timeout=timeout_s)
    except FuturesTimeoutError as exc:
        timed_out = True
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"Codur run exceeded {timeout_s} seconds") from exc
    finally:
        if not timed_out:
            executor.shutdown(wait=True, cancel_futures=True)


def _run_prompt(
    prompt: str,
    config: Optional[Path],
    verbose: bool,
    raw: bool,
    max_llm_calls: int | None,
    fail_early: bool,
    dump_messages: str | None = None,
) -> None:
    if not raw:
        console.print(Panel.fit(
            "[bold cyan]Codur Agent[/bold cyan]\n"
            f"Task: {prompt}",
            border_style="cyan"
        ))

    if fail_early:
        os.environ["EARLY_FAILURE_HELPERS_FOR_TESTS"] = "1"

    cfg = load_config(config)
    if max_llm_calls is not None:
        cfg.runtime.max_llm_calls = max_llm_calls
    graph = create_agent_graph(cfg)

    try:
        result = _invoke_graph(graph, {
            "messages": [HumanMessage(content=prompt)],
            "verbose": verbose,
            "config": cfg,
            "llm_calls": 0,
            "max_llm_calls": cfg.runtime.max_llm_calls,
        }, cfg.runtime.max_runtime_s)

        if dump_messages:
            with open(dump_messages, "w", encoding="utf-8") as f:
                for message in result.get("messages", []):
                    f.write("-" * 20 + f"{message.__class__.__name__}" + "-" * 20 + "\n\n")
                    f.write(f"{message.content}\n\n")

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
    max_llm_calls: Optional[int] = typer.Option(
        None,
        "--max-llm-calls",
        help="Maximum number of LLM calls for a single run",
    ),
    fail_early: bool = typer.Option(
        False,
        "--fail-early",
        help="Enable early failure helpers for tests",
    ),
    dump_messages: Optional[str] = typer.Option(
        None,
        "--dump-messages",
        help="Dump all messages to a file",
    ),
):
    if command:
        _run_prompt(command, config, verbose, raw, max_llm_calls, fail_early, dump_messages)
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
    max_llm_calls: Optional[int] = typer.Option(
        None,
        "--max-llm-calls",
        help="Maximum number of LLM calls for a single run",
    ),
    fail_early: bool = typer.Option(
        False,
        "--fail-early",
        help="Enable early failure helpers for tests",
    ),
    dump_messages: Optional[str] = typer.Option(
        None,
        "--dump-messages",
        help="Dump all messages to a file",
    ),
):
    """
    Run a coding task through the agent orchestrator.

    Example:
        codur run "Create a Python function to calculate fibonacci numbers"
        codur run "Refactor the authentication module" --verbose
    """
    _run_prompt(prompt, config, verbose, raw, max_llm_calls, fail_early, dump_messages)


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
def list_agents(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
):
    """List available agents and their capabilities."""
    from codur.agents import AgentRegistry
    # Ensure agents are registered
    import codur.graph.execution  # noqa: F401

    cfg = load_config(config)

    console.print("[bold cyan]Agent Information[/bold cyan]\n")

    # 1. Registered Agent Classes (Implementations)
    console.print("[bold yellow]Registered Agent Classes (Implementations):[/bold yellow]")
    agent_names = AgentRegistry.list_agents()
    if agent_names:
        for name in sorted(agent_names):
            agent_class = AgentRegistry.get(name)
            if agent_class:
                description = agent_class.get_description()
                console.print(f"[green]●[/green] [bold]{name}[/bold]")
                console.print(f"  {description}\n")
    else:
        console.print("  No class-based agents registered.\n")

    # 2. Configured Agent Profiles
    console.print("[bold yellow]Configured Agent Profiles (from codur.yaml):[/bold yellow]")
    configured_agents = cfg.agents.configs
    if configured_agents:
        for name, agent_cfg in sorted(configured_agents.items()):
            agent_type = getattr(agent_cfg, "type", "unknown")
            if agent_type == "llm":
                model = agent_cfg.config.get("model", "unknown")
                console.print(f"[green]●[/green] [bold]{name}[/bold] (LLM)")
                console.print(f"  Model: {model}")
                if "system_prompt" in agent_cfg.config:
                    prompt = agent_cfg.config['system_prompt']
                    if len(prompt) > 100:
                        prompt = prompt[:97] + "..."
                    console.print(f"  [dim]Prompt: {prompt}[/dim]")
                console.print()
            elif agent_type == "mcp":
                mcp_server = agent_cfg.config.get("mcp_server", "unknown")
                console.print(f"[green]●[/green] [bold]{name}[/bold] (MCP)")
                console.print(f"  MCP Server: {mcp_server}\n")
            elif agent_type == "tool":
                impl_name = name
                if name not in agent_names and getattr(agent_cfg, "name", None) in agent_names:
                    impl_name = agent_cfg.name
                
                console.print(f"[green]●[/green] [bold]{name}[/bold] (Tool)")
                if impl_name in agent_names:
                    console.print(f"  Implementation: {impl_name}")
                model = agent_cfg.config.get("model")
                if model:
                    console.print(f"  Model: {model}")
                console.print()
            else:
                console.print(f"[green]●[/green] [bold]{name}[/bold] ({agent_type})\n")
    else:
        console.print("  No agents configured in codur.yaml.\n")


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
def interactive(
    max_llm_calls: Optional[int] = typer.Option(
        None,
        "--max-llm-calls",
        help="Maximum number of LLM calls per prompt",
    ),
):
    """
    Start an interactive session with the coding agent.
    """
    console.print(Panel.fit(
        "[bold cyan]Codur Interactive Mode[/bold cyan]\n"
        "Type your tasks, or 'quit' to exit",
        border_style="cyan"
    ))

    cfg = load_config()
    if max_llm_calls is not None:
        cfg.runtime.max_llm_calls = max_llm_calls
    graph = create_agent_graph(cfg)

    while True:
        try:
            prompt = console.input("\n[bold cyan]codur>[/bold cyan] ")

            if prompt.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not prompt.strip():
                continue

            result = _invoke_graph(graph, {
                "messages": [HumanMessage(content=prompt)],
                "config": cfg,
                "llm_calls": 0,
                "max_llm_calls": cfg.runtime.max_llm_calls,
                "verbose": verbose,
            }, cfg.runtime.max_runtime_s)

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
