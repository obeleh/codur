"""
Main LangGraph definition for the coding agent
"""

from langgraph.graph import StateGraph, END

from codur.graph.state import AgentState
from codur.graph.execution import (
    execute_node,
    delegate_node,
    review_node,
)
from codur.graph.tools import tool_node
from codur.graph.coding_agent import coding_node
from codur.graph.explaining import explaining_node
from codur.graph.routing import should_continue, should_delegate
from codur.graph.planning.core import PlanningOrchestrator
from codur.graph.planning.phases.pattern_phase import pattern_plan
from codur.graph.planning.phases.llm_classification_phase import llm_classification
from codur.config import CodurConfig
from codur.constants import (
    AGENT_CODING,
    AGENT_EXPLAINING,
    REF_AGENT_CODING,
    REF_AGENT_EXPLAINING,
    ACTION_DELEGATE,
    ACTION_TOOL,
)
from codur.llm import create_llm, create_llm_profile
from codur.graph.state_operations import get_next_action, get_selected_agent


def should_continue_to_llm_classification(state: AgentState) -> str:
    """Route from pattern_plan to llm_classification."""
    next_action = get_next_action(state)
    if next_action == "continue_to_llm_classification":
        return "llm_classification"
    # If resolved in pattern_plan, route based on the decision
    return _route_based_on_decision(state)


def should_continue_to_llm_plan(state: AgentState) -> str:
    """Route from llm-classification to llm-plan."""
    next_action = get_next_action(state)
    if next_action == "continue_to_llm_plan":
        return "llm_plan"
    # If resolved in llm-classification, route based on the decision
    return _route_based_on_decision(state)


def _route_based_on_decision(state: AgentState) -> str:
    """Route based on the planning decision (delegate, tool, or end)."""
    next_action = get_next_action(state)
    if next_action == ACTION_DELEGATE:
        selected_agent = get_selected_agent(state)
        if selected_agent in (REF_AGENT_CODING, AGENT_CODING):
            return "coding"
        if selected_agent in (REF_AGENT_EXPLAINING, AGENT_EXPLAINING):
            return "explaining"
        return "delegate"
    elif next_action == ACTION_TOOL:
        return "tool"
    else:
        return "end"


def create_agent_graph(config: CodurConfig):
    """
    Create the main agent orchestration graph.

    Flow:
    1. Plan: Analyze task and decide approach
    2. Delegate: Route to appropriate agent/tool (Ollama, Codex, MCP server)
    3. Execute: Run the delegated action
    4. Review: Check result and decide if done
    5. Loop or finish
    """

    # Initialize LLM with JSON mode enabled for structured planning output
    # JSON mode forces the LLM to return valid JSON, improving reliability
    if config.llm.default_profile:
        llm = create_llm_profile(config, config.llm.default_profile, json_mode=True)
    else:
        llm = create_llm(config)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes - Three-phase planning architecture
    # Phase 0: Pattern-based classification and discovery
    workflow.add_node("pattern_plan", lambda state: pattern_plan(state, config))
    # Phase 1: LLM-based classification (config-gated, enabled by default)
    workflow.add_node("llm_classification", lambda state: llm_classification(state, config))
    # Phase 2: Full LLM planning
    workflow.add_node("llm_plan", lambda state: PlanningOrchestrator(config).llm_plan(state, llm))
    workflow.add_node("delegate", lambda state: delegate_node(state, config))
    workflow.add_node("tool", lambda state: tool_node(state, config))
    workflow.add_node("coding", lambda state: coding_node(state, config))
    workflow.add_node("explaining", lambda state: explaining_node(state, config))
    workflow.add_node("execute", lambda state: execute_node(state, config))
    workflow.add_node("review", lambda state: review_node(state, llm, config))

    # Set entry point to first planning phase
    workflow.set_entry_point("pattern_plan")

    # Phase transitions: pattern_plan → llm_classification (optional) → llm_plan
    workflow.add_conditional_edges(
        "pattern_plan",
        should_continue_to_llm_classification,
        {
            "llm_classification": "llm_classification",
            "delegate": "delegate",
            "tool": "tool",
            "coding": "coding",
            "explaining": "explaining",
            "end": END,
        }
    )

    workflow.add_conditional_edges(
        "llm_classification",
        should_continue_to_llm_plan,
        {
            "llm_plan": "llm_plan",
            "delegate": "delegate",
            "tool": "tool",
            "coding": "coding",
            "explaining": "explaining",
            "end": END,
        }
    )

    # Final planning phase routes to execution
    workflow.add_conditional_edges(
        "llm_plan",
        should_delegate,
        {
            "delegate": "delegate",
            "tool": "tool",
            "coding": "coding",
            "explaining": "explaining",
            "end": END,
        }
    )

    # Execution flow
    workflow.add_edge("delegate", "execute")
    workflow.add_edge("execute", "review")
    workflow.add_edge("tool", "review")
    workflow.add_edge("coding", "review")
    workflow.add_edge("explaining", "review")

    # Review loop - on retry, go directly to full planning (Phase 2) since we already classified
    # This avoids re-running Phases 0 and 1 which are not necessary on retries
    workflow.add_conditional_edges(
        "review",
        should_continue,
        {
            "continue": "llm_plan",  # Skip to Phase 2 for retries (we already have classification)
            "coding": "coding",
            "explaining": "explaining",
            "end": END,
        }
    )

    # Compile the graph with increased recursion limit for trial-error loops
    # Initial path (LLM classification disabled):
    #   pattern_plan → llm_plan → delegate → execute → review (5 nodes)
    # Initial path (LLM classification enabled):
    #   pattern_plan → llm_classification → llm_plan → delegate → execute → review (6 nodes)
    # Retries: llm_plan → delegate → execute → review → continue (4 nodes per retry)
    # With max_iterations=10 and max_tool_iterations=5 per agent:
    # Estimate: 6 (initial) + 10 * 4 (retries) + (5 tool iterations * 2) = 66 nodes
    # Using 350 for comprehensive trial-error loops with optimized retry path
    try:
        return workflow.compile(recursion_limit=350)
    except TypeError:
        return workflow.compile()
