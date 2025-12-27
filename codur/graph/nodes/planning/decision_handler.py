"""Decision parsing and handling for planning."""

from __future__ import annotations

from typing import Dict, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage

from codur.config import CodurConfig
from codur.graph.nodes.types import PlanNodeResult, PlanningDecision
from codur.graph.state import AgentState
from codur.utils.llm_calls import invoke_llm
from codur.utils.validation import require_config

from .json_parser import JSONResponseParser


class PlanningDecisionHandler:
    def __init__(self, config: CodurConfig) -> None:
        self.config = config
        self.parser = JSONResponseParser()

    def create_llm_debug(
        self,
        system_prompt: str,
        user_message: str,
        llm_response: str,
        retry_response: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> Dict[str, Any]:
        debug_short = self.config.planning.debug_truncate_short
        debug_long = self.config.planning.debug_truncate_long
        debug_info = {
            "node": "plan",
            "system_prompt": system_prompt[:debug_short] + "..."
            if len(system_prompt) > debug_short
            else system_prompt,
            "user_message": user_message,
            "llm_response": (
                llm_response[:debug_long] + "..."
                if len(llm_response) > debug_long
                else llm_response
            ),
        }

        if llm:
            model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
            debug_info["llm_model"] = model_name

        profile_name = self.config.llm.default_profile
        debug_info["llm_profile"] = profile_name

        if retry_response:
            debug_info["llm_response_retry"] = (
                retry_response[:debug_long] + "..."
                if len(retry_response) > debug_long
                else retry_response
            )
        return debug_info

    def parse_planning_response(
        self,
        llm: BaseChatModel,
        content: str,
        messages: list[BaseMessage],
        has_tool_results: bool,
        llm_debug: Dict[str, Any],
        state: AgentState | None = None,
    ) -> Optional[PlanningDecision]:
        decision = self.parser.parse(content)
        if decision:
            return decision

        if has_tool_results:
            retry_prompt = SystemMessage(
                content=(
                    "IMPORTANT: You must return ONLY a valid JSON object in your response. "
                    "If tools are needed next, use action 'tool' with tool_calls in JSON format. "
                    "If you can answer now, use action 'respond' with a concise response in JSON format."
                )
            )
            retry_response = invoke_llm(
                llm,
                [retry_prompt] + list(messages),
                invoked_by="planning.retry_json",
                state=state,
                config=self.config,
            )
            retry_content = retry_response.content
            debug_long = self.config.planning.debug_truncate_long
            llm_debug["llm_response_retry"] = (
                retry_content[:debug_long] + "..."
                if len(retry_content) > debug_long
                else retry_content
            )

            decision = self.parser.parse(retry_content)
            if decision:
                return decision

        return None

    def handle_decision(
        self,
        decision: PlanningDecision,
        iterations: int,
        llm_debug: Dict[str, Any],
    ) -> PlanNodeResult:
        action = decision.get("action", "delegate")

        if action == "respond":
            response_text = decision.get("response") or decision.get("reasoning", "Task completed")
            return {
                "next_action": "end",
                "final_response": response_text,
                "iterations": iterations + 1,
                "llm_debug": llm_debug,
            }
        if action == "tool":
            return {
                "next_action": "tool",
                "tool_calls": decision.get("tool_calls", []),
                "selected_agent": decision.get("agent"),
                "iterations": iterations + 1,
                "llm_debug": llm_debug,
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
                "llm_debug": llm_debug,
            }

        return {
            "next_action": "end",
            "final_response": "Task completed",
            "iterations": iterations + 1,
            "llm_debug": llm_debug,
        }
