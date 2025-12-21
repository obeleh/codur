"""Planning node package."""

from .core import PlanningOrchestrator


def plan_node(state, llm, config):
    return PlanningOrchestrator(config).plan(state, llm)


__all__ = ["plan_node", "PlanningOrchestrator"]
