"""Planning node package."""

from .core import PlanningOrchestrator, textual_pre_plan, llm_pre_plan


def plan_node(state, llm, config):
    """Deprecated: use textual_pre_plan, llm_pre_plan, or llm_plan instead."""
    return PlanningOrchestrator(config).llm_plan(state, llm)


def textual_pre_plan_node(state, config):
    """Phase 0: Textual (pattern-based) pre-planning."""
    return textual_pre_plan(state, config)


def llm_pre_plan_node(state, config):
    """Phase 1: LLM-based quick classification."""
    return llm_pre_plan(state, config)


def llm_plan_node(state, llm, config):
    """Phase 2: Full LLM planning for uncertain cases."""
    return PlanningOrchestrator(config).llm_plan(state, llm)


__all__ = [
    "plan_node",
    "textual_pre_plan_node",
    "llm_pre_plan_node",
    "llm_plan_node",
    "PlanningOrchestrator",
    "textual_pre_plan",
    "llm_pre_plan",
]
