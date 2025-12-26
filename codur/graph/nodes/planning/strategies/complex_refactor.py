"""Complex refactor strategy."""

from rich.console import Console
from langchain_core.messages import BaseMessage, HumanMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from codur.graph.nodes.planning.strategies.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    select_example_files,
    build_example_line,
    format_examples,
    normalize_agent_name,
    format_tool_suggestions,
)

console = Console()

# Domain-specific patterns for complex refactor tasks
_COMPLEX_REFACTOR_PATTERNS = PatternConfig(
    primary_keywords=frozenset({
        "refactor", "redesign", "migrate", "restructure", "rewrite",
        "multiple files", "entire", "all files", "codebase"
    }),
    boosting_keywords=frozenset({
        "architecture", "pattern", "service", "layer", "component",
        "module", "package", "namespace", "interface", "abstract"
    }),
    negative_keywords=frozenset({
        "weather", "news", "search", "google", "price", "stock"
    }),
    file_extensions=frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php",
        ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".kt"
    }),
    phrases=frozenset({
        "multiple files", "all files", "entire codebase"
    }),
)


class ComplexRefactorStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return patterns for complex refactor classification."""
        return _COMPLEX_REFACTOR_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute complex refactor classification score."""
        result = ScoreContribution(score=0.0)
        patterns = self.get_patterns()

        # Check for complex refactor keywords
        has_complex = any(kw in text_lower for kw in patterns.primary_keywords)
        if has_complex:
            # Reduce score if it looks like explanation or lookup
            explain_or_lookup_keywords = {
                "find", "search", "locate", "usage", "used", "explain", "describe",
                "summarize", "summary", "what does", "how does", "tell me about"
            }
            has_explain_or_lookup = any(kw in text_lower for kw in explain_or_lookup_keywords)
            base_score = 0.35 if has_explain_or_lookup else 0.7
            result.add(base_score, "complex refactor keyword")
            if len(detected_files) > 1:
                result.add(0.1, "multiple files hinted")

        # Multi-file refactor cues
        refactor_language_keywords = {
            "behavior", "logic", "rule", "rules", "function", "method", "class", "module",
            "code", "implementation", "cache", "caching", "log", "logs", "logging",
            "validation", "parsing", "processing", "handler", "workflow", "flow", "pipeline",
        }
        refactor_language = any(kw in text_lower for kw in refactor_language_keywords)
        if len(detected_files) > 1 and refactor_language:
            result.add(0.75, "multi-file refactor cues")

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

        # Complex refactoring is too risky to delegate from Phase 0
        # Pass to Phase 2 for careful analysis and routing
        if verbose and classification.is_confident:
            console.print("[dim]Complex refactor detected, deferring to Phase 2 for analysis[/dim]")

        return None

    def build_planning_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        suggested_tools = format_tool_suggestions([
            "file_tree",
            "list_dirs",
            "python_dependency_graph",
            "python_ast_dependencies_multifile",
            "code_quality",
            "lint_python_tree",
            "search_files",
            "grep_files",
            "ripgrep_search",
        ])
        default_agent = normalize_agent_name(
            config.agents.preferences.routing.get("multifile"),
            normalize_agent_name(
                config.agents.preferences.default_agent,
                "agent:codur-coding",
            ),
        )
        first_file, second_file = select_example_files(classification.detected_files)
        examples = [
            build_example_line(
                "Refactor the project to use a service layer",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "discover files for multi-file refactor",
                    "response": None,
                    "tool_calls": [{"tool": "list_files", "args": {}}],
                },
            ),
            build_example_line(
                f"Refactor {first_file} and {second_file} to share a base class",
                {
                    "action": "delegate",
                    "agent": default_agent,
                    "reasoning": "multi-file refactor",
                    "response": None,
                    "tool_calls": [],
                },
            ),
        ]
        focus = (
            "**Task Focus: Complex Refactor**\n"
            "- This likely spans multiple files; consider list_files if no hints are present.\n"
            "- Prefer multi-file capable agents when routing.\n"
            f"- {suggested_tools}\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
