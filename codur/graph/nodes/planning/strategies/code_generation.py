"""Code generation strategy."""

from langchain_core.messages import BaseMessage, HumanMessage

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
    normalize_agent_name,
    format_tool_suggestions,
)
from codur.graph.nodes.planning.injectors import get_injector_for_file

# Domain-specific patterns for code generation tasks
_CODE_GENERATION_PATTERNS = PatternConfig(
    primary_keywords=frozenset({
        "write", "create", "add", "generate", "make", "build", "new",
        "implement", "complete", "finish", "solve"
    }),
    boosting_keywords=frozenset({
        "function", "class", "module", "script", "cli", "tool", "automation",
        "dashboard", "ui", "interface", "api", "service", "app", "workflow",
        "pipeline", "code", "program", "library"
    }),
    negative_keywords=frozenset({
        "fix", "bug", "error", "debug", "broken", "wrong",
        "weather", "news", "search", "google", "price", "stock"
    }),
    file_extensions=frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php",
        ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".kt"
    }),
)


class CodeGenerationStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return patterns for code generation classification."""
        return _CODE_GENERATION_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute code generation classification score."""
        result = ScoreContribution(score=0.0)
        patterns = self.get_patterns()

        # Check for generation keywords
        has_generation = any(kw in text_lower for kw in patterns.primary_keywords)
        if has_generation:
            result.add(0.6, "generation keyword")
            if detected_files:
                result.add(0.1, "file hint present")
            if "docstring" in text_lower or "requirements" in text_lower:
                result.add(0.05, "explicit requirements")
            # Boost if no fix keywords
            if not any(kw in text_lower for kw in patterns.negative_keywords):
                result.add(0.05, "no fix keywords")
            # Boost if no file hint (pure generation)
            if not detected_files:
                result.add(0.05, "no file hint")
            # Boost for code artifact context
            strong_code_context = has_code_file or any(
                kw in text_lower for kw in patterns.boosting_keywords
            )
            if strong_code_context:
                result.add(0.15, "code artifact context")

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
        """Execute file discovery for code generation tasks."""
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
            "write_file",
            "append_file",
            "inject_function",
            "inject_lines",
            "replace_lines",
            "python_ast_outline",
            "read_json",
            "read_yaml",
            "read_ini",
        ])
        example_path = select_example_file(classification.detected_files)
        # Use injector to get language-specific example tool calls
        injector = get_injector_for_file(example_path)
        if injector:
            example_tool_calls = injector.get_example_tool_calls(example_path)
        else:
            example_tool_calls = [{"tool": "read_file", "args": {"path": example_path}}]
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
            "- If a file path is known, call read_file first (language-specific tools auto-injected via Tool Injectors).\n"
            "- If no file path is known, delegate to a generation-capable agent.\n"
            "- Prefer agent:codur-coding for coding challenges with docstrings/requirements.\n"
            f"- {suggested_tools}\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
