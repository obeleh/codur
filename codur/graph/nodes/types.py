"""Shared type definitions for graph nodes."""

from typing import Dict, Any, TypedDict, NotRequired, Literal, Optional
from langchain_core.messages import BaseMessage


class ToolCall(TypedDict):
    tool: str
    args: Dict[str, Any]


class AgentOutcome(TypedDict):
    agent: str
    result: NotRequired[str]
    status: str


class PlanNodeResult(TypedDict):
    next_action: Literal["end", "delegate", "tool"]
    iterations: int
    final_response: NotRequired[str]
    selected_agent: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]
    llm_debug: NotRequired[Dict[str, Any]]


class DelegateNodeResult(TypedDict):
    agent_outcome: AgentOutcome


class ToolNodeResult(TypedDict):
    agent_outcome: AgentOutcome
    messages: list[BaseMessage]


class ExecuteNodeResult(TypedDict):
    agent_outcome: AgentOutcome


class ReviewNodeResult(TypedDict):
    final_response: str
    next_action: Literal["end"]


class PlanningDecision(TypedDict):
    action: Literal["delegate", "respond", "tool", "done"]
    agent: NotRequired[Optional[str]]
    reasoning: NotRequired[str]
    response: NotRequired[Optional[str]]
    tool_calls: NotRequired[list[ToolCall]]
