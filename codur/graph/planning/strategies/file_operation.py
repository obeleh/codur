"""File operation strategy."""

from rich.console import Console
from langchain_core.messages import BaseMessage

from codur.graph.node_types import PlanNodeResult
from codur.graph.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from codur.graph.planning.strategies.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    select_example_files,
    build_example_line,
    format_examples,
    format_tool_suggestions,
)

console = Console()

# Domain-specific patterns for file operation tasks
_FILE_OPERATION_PATTERNS = PatternConfig(
    primary_keywords=frozenset({
        "move", "copy", "delete", "remove", "rename"
    }),
    boosting_keywords=frozenset({
        "file", "files", "directory", "folder", "path"
    }),
    negative_keywords=frozenset({
        "fix", "bug", "error", "refactor", "rewrite", "implement",
        "function", "method", "class", "code", "logic"
    }),
    file_extensions=frozenset({
        ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".txt", ".md",
        ".html", ".css", ".xml", ".csv"
    }),
    action_keywords={
        "move": "move_file",
        "copy": "copy_file",
        "delete": "delete_file",
        "remove": "delete_file",
        "rename": "move_file",
    },
    phrases=frozenset({
        "list files", "list all files", "show files", "show all files", "list directory"
    }),
)


def _contains_word(text: str, word: str) -> bool:
    """Check if word appears as a whole word in text."""
    import re
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


class FileOperationStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return patterns for file operation classification."""
        return _FILE_OPERATION_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute file operation classification score."""
        result = ScoreContribution(score=0.0)
        patterns = self.get_patterns()

        # Check for file listing intent (highest confidence)
        for phrase in patterns.phrases:
            if phrase in text_lower:
                result.add(0.85, "file listing intent")
                return result  # Early return for file listing

        # Check for file operation keywords with files
        if not detected_files:
            return result

        # Code context keywords reduce file operation score
        code_context_keywords = {
            "behavior", "logic", "rule", "rules", "function", "method", "class", "module",
            "code", "implementation", "cache", "caching", "log", "logs", "logging",
            "validation", "parsing", "processing", "handler", "workflow", "flow", "pipeline",
        }
        has_code_context = has_code_file or any(kw in text_lower for kw in code_context_keywords)
        refactor_language = any(kw in text_lower for kw in code_context_keywords)

        for keyword in patterns.action_keywords:
            if not _contains_word(text_lower, keyword):
                continue

            # Start with high base score
            penalty = 0.0

            # Apply penalties for code context
            if has_code_context:
                penalty += 0.4
            if any(kw in text_lower for kw in {"fix", "bug", "error", "create", "write", "explain"}):
                penalty += 0.2
            if keyword in ("rename", "remove") and has_code_context:
                penalty += 0.2
            if refactor_language:
                penalty += 0.35

            # Reduce penalty for non-code files
            if detected_files and not has_code_file:
                penalty = max(0.0, penalty - 0.4)

            score = max(0.0, 0.9 - penalty)
            result.add(score, f"file operation keyword: {keyword}")

        return result

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

    def build_planning_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        suggested_tools = format_tool_suggestions([
            "list_dirs",
            "file_tree",
            "read_file",
        ])
        source_path, destination_path = select_example_files(classification.detected_files)
        examples = [
            build_example_line(
                f"delete {source_path}",
                {
                    "action": "delegate",
                    "agent": "agent:codur-coding",
                },
            ),
            build_example_line(
                f"copy {source_path} to {destination_path}",
                {
                    "action": "delegate",
                    "agent": "agent:codur-coding",
                },
            ),
        ]
        focus = (
            "**Task Focus: File Operation**\n"
            "- File operations should be delegated to the coding agent\n"
            "- Use delegate_task(\"agent:codur-coding\", \"<file operation instructions>\")\n"
            f"- {suggested_tools}\n"
            "\n"
            "Examples:\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
