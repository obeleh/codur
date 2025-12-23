"""Web search strategy for Phase 1."""

from rich.console import Console
from langchain_core.messages import BaseMessage, HumanMessage

from codur.graph.nodes.types import PlanNodeResult
from codur.graph.nodes.planning.types import ClassificationResult
from codur.config import CodurConfig
from codur.graph.nodes.planning.hints.prompt_utils import (
    build_base_prompt,
    format_focus_prompt,
    build_example_line,
    format_examples,
)

console = Console()

class WebSearchStrategy:
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

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
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
            "- Return ONLY a valid JSON object.\n"
            "Examples (context-aware):\n"
            f"{format_examples(examples)}"
        )
        return format_focus_prompt(build_base_prompt(config), focus, classification.detected_files)
