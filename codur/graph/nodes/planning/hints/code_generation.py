"""Code generation strategy for Phase 1."""

from rich.console import Console
from langchain_core.messages import BaseMessage, HumanMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
from codur.graph.nodes.planning.agent_selection import select_agent_for_task
from codur.config import CodurConfig
from codur.graph.nodes.planning.tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
)
from codur.graph.nodes.planning.hints.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    select_example_file,
    build_example_line,
    format_examples,
    normalize_agent_name,
)

console = Console()

class CodeGenerationStrategy:
    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:
        
        # 1. No file hint -> list files (discovery)
        if not tool_results_present and not classification.detected_files:
            if verbose:
                console.print("[dim]No file hint detected - listing files[/dim]")
            return {
                "next_action": "tool",
                "tool_calls": [{"tool": "list_files", "args": {}}],
                "iterations": iterations + 1,
                "llm_debug": {
                    "phase1_resolved": True,
                    "task_type": classification.task_type.value,
                    "file_discovery": "list_files",
                },
            }

        # 2. Tool results present -> read file
        if tool_results_present and not classification.detected_files and not tool_results_include_read_file(messages):
            candidate = select_file_from_tool_results(messages)
            if candidate:
                if verbose:
                    console.print(f"[dim]Selected file from tool results: {candidate}[/dim]")
                return {
                    "next_action": "tool",
                    "tool_calls": [{"tool": "read_file", "args": {"path": candidate}}],
                    "iterations": iterations + 1,
                    "llm_debug": {
                        "phase1_resolved": True,
                        "task_type": classification.task_type.value,
                        "file_discovery": candidate,
                    },
                }

        # 3. High confidence -> delegate
        if classification.is_confident and not tool_results_present:
            user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
            agent = select_agent_for_task(
                config=config,
                user_message=user_message,
                detected_files=classification.detected_files,
                routing_key="simple",
                prefer_multifile=len(classification.detected_files) > 1,
                allow_coding_agent=True,
            )
            
            # Standard delegation for generation
            if verbose:
                console.print("[green]✓ Code generation resolved (high confidence)[/green]")
            return {
                "next_action": "delegate",
                "selected_agent": agent,
                "iterations": iterations + 1,
                "llm_debug": {
                    "phase1_resolved": True,
                    "task_type": classification.task_type.value,
                    "selected_agent": agent
                },
            }

        return None

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        example_path = select_example_file(classification.detected_files)
        example_tool_calls = [{"tool": "read_file", "args": {"path": example_path}}]
        if example_path.endswith(".py"):
            example_tool_calls.append(
                {"tool": "python_ast_dependencies", "args": {"path": example_path}}
            )
        default_agent = normalize_agent_name(
            config.agents.preferences.default_agent,
            "agent:codur-coding",
        )
        examples = [
            build_example_line(
                f"Implement title case in {example_path}",
                {
                    "action": "tool",
                    "agent": "agent:codur-coding",
                    "reasoning": "read file to get docstring and context",
                    "response": None,
                    "tool_calls": example_tool_calls,
                },
            ),
            build_example_line(
                "Write a sorting function",
                {
                    "action": "delegate",
                    "agent": default_agent,
                    "reasoning": "code generation request",
                    "response": None,
                    "tool_calls": [],
                },
            ),
        ]
        focus = (
            "**Task Focus: Code Generation**\n"
            "- If a file path is known, call read_file first (python files auto-trigger AST deps).\n"
            "- If no file path is known, delegate to a generation-capable agent.\n"
            "- Prefer agent:codur-coding for coding challenges with docstrings/requirements.\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
