"""Planning orchestrator."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from codur.config import CodurConfig
from codur.graph.node_types import PlanNodeResult
from codur.graph.state import AgentState

from .prompt_builder import PlanningPromptBuilder
from .phases.plan_phase import llm_plan


class PlanningOrchestrator:
    def __init__(self, config: CodurConfig) -> None:
        self.config = config
        self.prompt_builder = PlanningPromptBuilder(config)
        # decision_handler and json_parser are no longer needed

    def llm_plan(self, state: AgentState, llm: BaseChatModel) -> PlanNodeResult:
        return llm_plan(
            config=self.config,
            prompt_builder=self.prompt_builder,
            decision_handler=None,  # Deprecated
            json_parser=None,  # Deprecated
            state=state,
            llm=None,  # Deprecated - created internally
        )
