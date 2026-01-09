"""Decision parsing and handling for planning."""

from __future__ import annotations

from codur.config import CodurConfig
from codur.graph.node_types import PlanNodeResult, PlanningDecision
from codur.utils.config_helpers import require_default_agent


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
            default_agent = require_default_agent(self.config)
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
