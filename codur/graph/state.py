"""
State definition for the agent graph
"""

from __future__ import annotations

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

from codur.config import CodurConfig


class AgentState(TypedDict):
    """
    State for the coding agent graph.

    Attributes:
        messages: The conversation history
        next_action: The next action to take
        agent_outcome: Result from the last agent action
        iterations: Number of iterations so far
        final_response: The final response to return
        selected_agent: The agent selected for delegation
        tool_calls: Tool calls requested by the planner
        verbose: Whether to print verbose output
        llm_calls: Count of LLM invocations so far
        max_llm_calls: Maximum allowed LLM invocations
        error_hashes: Hashes of errors seen for deduplication
        local_repair_attempted: Whether a local repair was attempted
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str
    agent_outcome: dict
    iterations: int
    final_response: str
    selected_agent: str
    tool_calls: list[dict]
    verbose: bool
    config: CodurConfig
    llm_calls: int
    max_llm_calls: int | None
    error_hashes: list
    local_repair_attempted: bool
    next_step_suggestion: str | None


class AgentStateData(dict):
    """Dict wrapper with helpers for tool access."""

    def get_config(self):
        return self.get("config")



