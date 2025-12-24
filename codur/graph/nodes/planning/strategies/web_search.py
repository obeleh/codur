"""Web search strategy."""

from rich.console import Console
from langchain_core.messages import BaseMessage, HumanMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult, PatternConfig, ScoreContribution
from codur.config import CodurConfig
from codur.graph.nodes.planning.strategies.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    build_example_line,
    format_examples,
    format_tool_suggestions,
)

console = Console()

# Domain-specific patterns for web search tasks
_WEB_SEARCH_PATTERNS = PatternConfig(
    primary_keywords=frozenset({
        "weather", "search", "latest", "news", "today", "current", "how is", "what are",
        "who is", "when did", "stock", "price", "market", "google", "find"
    }),
    boosting_keywords=frozenset({
        "weather", "news", "price", "stock", "market", "bitcoin", "forecast"
    }),
    # Code artifact keywords that indicate code generation, not web search
    # Note: These are specific code artifacts, not verbs like "create" or "write"
    negative_keywords=frozenset({
        "code", "script", "function", "class", "module", "api", "cli", "library",
        "program", "dashboard", "ui", "interface", "frontend", "backend"
    }),
    file_extensions=frozenset(),  # Web search doesn't involve local files
    phrases=frozenset({
        "how is", "what are", "who is", "when did"
    }),
)


class WebSearchStrategy:
    def get_patterns(self) -> PatternConfig:
        """Return patterns for web search classification."""
        return _WEB_SEARCH_PATTERNS

    def compute_score(
        self,
        text_lower: str,
        words: set[str],
        detected_files: list[str],
        has_code_file: bool,
    ) -> ScoreContribution:
        """Compute web search classification score."""
        result = ScoreContribution(score=0.0)
        patterns = self.get_patterns()

        # Check for web search keywords
        for keyword in patterns.primary_keywords:
            if keyword in text_lower:
                result.add(0.12, f"web keyword: {keyword}")
                if not detected_files and not has_code_file:
                    result.add(0.08, "no local code context")

        # Strong web keywords get extra boost
        for keyword in patterns.boosting_keywords:
            if keyword in text_lower:
                result.add(0.18, f"strong web keyword: {keyword}")

        # Web intent without code context
        has_web = any(kw in text_lower for kw in patterns.primary_keywords)
        has_code_context = has_code_file or any(
            kw in text_lower for kw in patterns.negative_keywords
        )
        if has_web and not detected_files and not has_code_context:
            result.add(0.25, "web intent without code context")

        # Broad questions without code context
        broad_question_words = {"who", "when", "where", "why", "what"}
        has_broad_question = bool(words & broad_question_words)
        if has_broad_question and not detected_files and not has_code_context:
            result.add(0.2, "broad question without code context")

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
        """Handle web search tasks by triggering search immediately."""
        # Only trigger if no tools have run yet
        if tool_results_present:
            return None

        user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
        
        if verbose:
            console.print("[dim]Web search task detected - triggering search[/dim]")
            
        return {
            "next_action": "tool",
            "tool_calls": [{"tool": "duckduckgo_search", "args": {"query": user_message}}],
            "iterations": iterations + 1,
            "llm_debug": {
                "phase1_resolved": True,
                "task_type": "web_search",
                "search_query": user_message,
            },
        }

    def build_planning_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        suggested_tools = format_tool_suggestions([
            "duckduckgo_search",
            "fetch_webpage",
            "location_lookup",
        ])
        examples = [
            build_example_line(
                "What is the current price of Bitcoin?",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "need real-time price info",
                    "response": None,
                    "tool_calls": [{"tool": "duckduckgo_search", "args": {"query": "current price of Bitcoin"}}],
                },
            ),
            build_example_line(
                "Summarize https://example.com/release-notes",
                {
                    "action": "tool",
                    "agent": None,
                    "reasoning": "need to fetch a specific URL",
                    "response": None,
                    "tool_calls": [
                        {
                            "tool": "fetch_webpage",
                            "args": {"url": "https://example.com/release-notes"},
                        }
                    ],
                },
            ),
        ]
        focus = (
            "**Task Focus: Web Search**\n"
            "- Use duckduckgo_search (or fetch_webpage if a URL is provided).\n"
            "- Do not respond without running the tool first.\n"
            f"- {suggested_tools}\n"
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
