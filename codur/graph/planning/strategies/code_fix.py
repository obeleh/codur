"""Code fix strategy."""

from langchain_core.messages import BaseMessage

from codur.graph.node_types import PlanNodeResult
from codur.graph.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from codur.graph.planning.strategies.discovery import discover_files_if_needed
from codur.graph.planning.strategies.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    select_example_file,
    build_example_line,
    format_examples,
    format_tool_suggestions,
)
from codur.graph.planning.injectors import get_injector_for_file

# Domain-specific patterns for code fix tasks
_CODE_FIX_PATTERNS = PatternConfig(
    primary_keywords=frozenset({
        "fix", "bug", "error", "debug", "issue", "broken", "incorrect", "wrong",
        "repair", "fail", "fails", "failed", "failing", "failure"
    }),
    boosting_keywords=frozenset({
        "traceback", "stack", "exception", "crash", "hang", "freeze",
        "not working", "doesn't work", "broken"
    }),
    negative_keywords=frozenset({
        "weather", "news", "search", "google", "price", "stock"
    }),
    file_extensions=frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php",
        ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".kt"
    }),
)


class CodeFixStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return patterns for code fix classification."""
        return _CODE_FIX_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute code fix classification score."""
        result = ScoreContribution(score=0.0)
        patterns = self.get_patterns()

        # Check for fix keywords
        has_fix = any(kw in text_lower for kw in patterns.primary_keywords)
        if has_fix:
            result.add(0.65, "fix/debug keyword")
            if detected_files:
                result.add(0.2, "file hint present")
            if has_code_file:
                result.add(0.1, "code context cues")
            # Web terms in fix context still get boost (user wants to fix something about web)
            web_terms = {"weather", "news", "search", "google", "price", "stock"}
            if any(kw in text_lower for kw in web_terms):
                result.add(0.2, "fix request with web terms")

        # File hint with code context but no explicit intent
        code_context_keywords = {
            "traceback", "stack", "exception", "test", "tests", "unit test",
            "log", "logging", "debug", "print", "printf", "function", "method", "class",
            "module", "import", "lint", "format", "typing", "type", "edge case",
            "script", "cli", "tool", "automation", "dashboard", "ui", "interface",
            "api", "service", "app", "workflow", "flow", "pipeline", "scheduler",
            "cron", "report", "reports", "frontend", "backend", "code",
        }
        has_code_context = has_code_file or any(kw in text_lower for kw in code_context_keywords)
        has_explicit_intent = any(
            any(kw in text_lower for kw in kw_set)
            for kw_set in [patterns.primary_keywords, {"write", "create", "explain", "refactor"}]
        )
        if detected_files and has_code_context and not has_explicit_intent:
            result.add(0.3, "file hint with code context")

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
        """Execute file discovery for code fix tasks."""
        return discover_files_if_needed(
            classification=classification,
            tool_results_present=tool_results_present,
            messages=messages,
            iterations=iterations,
            verbose=verbose,
            context_message="Reading {file_path} for context, routing decision in Phase 2",
        )

    def build_planning_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        suggested_tools = format_tool_suggestions([
            "search_files",
            "grep_files",
            "ripgrep_search",
            "python_ast_dependencies_multifile",
            "lint_python_files",
            "lint_python_tree",
            "validate_python_syntax",
            "code_quality",
            "python_dependency_graph",
        ])
        example_path = select_example_file(classification.detected_files)
        # Use injector to get language-specific example tool calls
        injector = get_injector_for_file(example_path)
        if injector:
            example_tool_calls = injector.get_example_tool_calls(example_path)
        else:
            example_tool_calls = [{"tool": "read_file", "args": {"path": example_path}}]
        examples = [
            build_example_line(
                f"Fix the bug in {example_path}",
                {
                    "action": "tool",
                    "agent": "agent:codur-coding",
                    "reasoning": "read file to get context for coding agent",
                    "response": None,
                    "tool_calls": example_tool_calls,
                },
            ),
            build_example_line(
                "Fix the failing tests",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "discover likely involved files",
                    "response": None,
                    "tool_calls": [{"tool": "list_files", "args": {}}],
                },
            ),
        ]
        focus = (
            "**Task Focus: Code Fix**\n"
            "- If a file path is known, call read_file first (language-specific tools auto-injected via Tool Injectors).\n"
            "- If no file path is known, call list_files to discover candidates, then read a likely file.\n"
            "- Prefer agent:codur-coding for coding challenges with docstrings/requirements.\n"
            f"- {suggested_tools}\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
