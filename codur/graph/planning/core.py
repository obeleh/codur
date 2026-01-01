"""Planning orchestrator."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from codur.config import CodurConfig
from codur.graph.node_types import PlanNodeResult
from codur.graph.state import AgentState
from codur.utils.json_parser import JSONResponseParser

from .decision_handler import PlanningDecisionHandler
from .prompt_builder import PlanningPromptBuilder
from .phases.plan_phase import llm_plan


class PlanningOrchestrator:
    def __init__(self, config: CodurConfig) -> None:
        self.config = config
        self.prompt_builder = PlanningPromptBuilder(config)
        self.decision_handler = PlanningDecisionHandler(config)
        self.json_parser = JSONResponseParser()

    def llm_plan(self, state: AgentState, llm: BaseChatModel) -> PlanNodeResult:
        return llm_plan(
            self.config,
            self.prompt_builder,
            self.decision_handler,
            self.json_parser,
            state,
            llm,
        )
