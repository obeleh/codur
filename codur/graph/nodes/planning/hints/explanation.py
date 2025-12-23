"""Explanation strategy for Phase 1."""

from rich.console import Console
from langchain_core.messages import BaseMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
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
)

console = Console()

class ExplanationStrategy:
    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:
        
        # 1. No file hint -> list files
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

        # 3. High confidence with files -> read file
        if classification.is_confident and not tool_results_present and classification.detected_files:
            if verbose:
                console.print("[green]✓ Explanation resolved (high confidence)[/green]")
            return {
                "next_action": "tool",
                "tool_calls": [{"tool": "read_file", "args": {"path": classification.detected_files[0]}}],
                "iterations": iterations + 1,
                "llm_debug": {"phase1_resolved": True, "task_type": "explanation"},
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
        examples = [
            build_example_line(
                f"What does {example_path} do?",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "read file for explanation",
                    "response": None,
                    "tool_calls": example_tool_calls,
                },
            ),
            build_example_line(
                "Explain how the project works",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "discover relevant files for explanation",
                    "response": None,
                    "tool_calls": [{"tool": "list_files", "args": {}}],
                },
            ),
        ]
        focus = (
            "**Task Focus: Explanation**\n"
            "- If a file path is known, call read_file first (python files auto-trigger AST deps).\n"
            "- If no file path is known, call list_files to discover candidates.\n"
            "- After tool results, respond with a concise explanation.\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
