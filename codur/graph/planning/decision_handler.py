"""Decision parsing and handling for planning."""

from __future__ import annotations

from typing import Dict, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from codur.config import CodurConfig
from codur.graph.node_types import PlanNodeResult, PlanningDecision
from codur.graph.state import AgentState
from codur.utils.llm_calls import invoke_llm
from codur.utils.validation import require_config
from codur.utils.json_parser import JSONResponseParser


class PlanningDecisionHandler:
    def __init__(self, config: CodurConfig) -> None:
        self.config = config

    def handle_decision(
        self,
        decision: PlanningDecision,
        iterations: int,
    ) -> PlanNodeResult:
        action = decision.get("action", "delegate")

        if action == "respond":
            response_text = decision.get("response") or decision.get("reasoning", "Task completed")
            return {
                "next_action": "end",
                "final_response": response_text,
                "iterations": iterations + 1,
            }
        if action == "tool":
            return {
                "next_action": "tool",
                "tool_calls": decision.get("tool_calls", []),
                "selected_agent": decision.get("agent"),
                "iterations": iterations + 1,
            }
        if action == "delegate":
            default_agent = self.config.agents.preferences.default_agent
            require_config(
                default_agent,
                "agents.preferences.default_agent",
                "agents.preferences.default_agent must be configured",
            )
            return {
                "next_action": "delegate",
                "selected_agent": decision.get("agent", default_agent),
                "iterations": iterations + 1,
            }

        return {
            "next_action": "end",
            "final_response": "Task completed",
            "iterations": iterations + 1,
        }
