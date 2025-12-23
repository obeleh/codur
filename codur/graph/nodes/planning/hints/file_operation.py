"""File operation strategy for Phase 1."""

from rich.console import Console
from langchain_core.messages import BaseMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
from codur.config import CodurConfig
from codur.graph.nodes.planning.hints.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    select_example_files,
    build_example_line,
    format_examples,
)

console = Console()

class FileOperationStrategy:
    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:
        """Handle file operations directly."""
        if tool_results_present:
            return None
            
        action = classification.detected_action
        if action and classification.detected_files:
            tool_call = {"tool": action, "args": {"path": classification.detected_files[0]}}
            # Handle move/copy which need source and destination
            if action in ("move_file", "copy_file") and len(classification.detected_files) >= 2:
                tool_call = {
                    "tool": action,
                    "args": {
                        "source": classification.detected_files[0],
                        "destination": classification.detected_files[1]
                    }
                }
            
            if verbose:
                console.print("[green]✓ File operation resolved (high confidence)[/green]")
                
            return {
                "next_action": "tool",
                "tool_calls": [tool_call],
                "iterations": iterations + 1,
                "llm_debug": {"phase1_resolved": True, "task_type": "file_operation"},
            }
            
        return None

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        source_path, destination_path = select_example_files(classification.detected_files)
        examples = [
            build_example_line(
                f"delete {source_path}",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "delete file",
                    "response": None,
                    "tool_calls": [{"tool": "delete_file", "args": {"path": source_path}}],
                },
            ),
            build_example_line(
                f"copy {source_path} to {destination_path}",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "copy file",
                    "response": None,
                    "tool_calls": [
                        {
                            "tool": "copy_file",
                            "args": {"source": source_path, "destination": destination_path},
                        }
                    ],
                },
            ),
        ]
        focus = (
            "**Task Focus: File Operation**\n"
            "- Use action: \"tool\" with the correct file operation tool.\n"
            "- Do not respond or delegate.\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
