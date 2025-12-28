"""Planning node package."""

from .core import PlanningOrchestrator, pattern_plan, pattern_plan, llm_pre_plan


def plan_node(state, llm, config):
    """Deprecated: use pattern_plan, llm_pre_plan, or llm_plan instead."""
    return PlanningOrchestrator(config).llm_plan(state, llm)


def pattern_plan_node(state, config):
    """Phase 0: Pattern-based pre-planning (merged textual + classification)."""
    return pattern_plan(state, config)


def textual_pre_plan_node(state, config):
    """Deprecated: use pattern_plan_node instead."""
    return pattern_plan(state, config)


def llm_pre_plan_node(state, config):
    """Phase 1: LLM-based classification (experimental, config-gated)."""
    return llm_pre_plan(state, config)


def llm_plan_node(state, llm, config):
    """Phase 2: Full LLM planning for uncertain cases."""
    return PlanningOrchestrator(config).llm_plan(state, llm)


__all__ = [
    "plan_node",
    "pattern_plan_node",
    "textual_pre_plan_node",
    "llm_pre_plan_node",
    "llm_plan_node",
    "PlanningOrchestrator",
    "pattern_plan",
    "pattern_plan",
    "llm_pre_plan",
]
