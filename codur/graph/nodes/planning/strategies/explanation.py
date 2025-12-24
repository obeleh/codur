"""Explanation strategy."""

from langchain_core.messages import BaseMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from codur.graph.nodes.planning.strategies.discovery import discover_files_if_needed
from codur.graph.nodes.planning.strategies.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    select_example_file,
    build_example_line,
    format_examples,
)

# Domain-specific patterns for explanation tasks
_EXPLANATION_PATTERNS = PatternConfig(
    primary_keywords=frozenset({
        "what does", "explain", "describe", "how does", "tell me about",
        "what is", "summarize", "summary", "how to use"
    }),
    boosting_keywords=frozenset({
        "project", "codebase", "repo", "repository", "code", "function",
        "method", "class", "module"
    }),
    negative_keywords=frozenset({
        "fix", "create", "write", "generate", "make", "build",
        "weather", "news", "price", "stock"
    }),
    file_extensions=frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php",
        ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".kt", ".md", ".txt"
    }),
    phrases=frozenset({
        "what does", "how does", "tell me about", "what is", "how to use"
    }),
)


class ExplanationStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return patterns for explanation classification."""
        return _EXPLANATION_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute explanation classification score."""
        result = ScoreContribution(score=0.0)
        patterns = self.get_patterns()

        # Check for explanation keywords
        has_explain = any(kw in text_lower for kw in patterns.primary_keywords)
        if has_explain:
            result.add(0.55, "explanation keyword")
            if detected_files:
                result.add(0.2, "file hint present")
            # Broad question with code context
            broad_question_words = {"who", "when", "where", "why", "what"}
            has_broad_question = bool(words & broad_question_words)
            if has_broad_question and has_code_file:
                result.add(0.1, "question about code context")

        # Lookup intent with project/code context
        lookup_keywords = {"find", "search", "locate", "usage", "used", "reference", "references", "where", "who", "when"}
        lookup_intent = any(kw in text_lower for kw in lookup_keywords)
        project_hint = any(kw in text_lower for kw in ("project", "codebase", "repo", "repository"))
        if lookup_intent and (project_hint or has_code_file or detected_files):
            result.add(0.45, "code lookup request")
        elif lookup_intent and not any(kw in text_lower for kw in {"weather", "news", "price", "stock"}):
            result.add(0.3, "lookup question without web intent")

        # Broad question with code context but no explicit explain keyword
        broad_question_words = {"who", "when", "where", "why", "what"}
        has_broad_question = bool(words & broad_question_words)
        code_context_keywords = {
            "code", "function", "method", "class", "module", "api", "interface",
        }
        has_code_context = has_code_file or any(kw in text_lower for kw in code_context_keywords)
        if has_broad_question and has_code_context and not has_explain:
            result.add(0.35, "broad question with code context")

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
        """Execute file discovery for explanation tasks."""
        return discover_files_if_needed(
            classification=classification,
            tool_results_present=tool_results_present,
            messages=messages,
            iterations=iterations,
            verbose=verbose,
        )

    def build_planning_prompt(
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
