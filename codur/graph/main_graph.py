"""
Main LangGraph definition for the coding agent
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rich.console import Console

from codur.graph.state import AgentState
from codur.graph.nodes import (
    plan_node,
    execute_node,
    delegate_node,
    tool_node,
    review_node,
    should_continue,
    should_delegate,
)
from codur.config import CodurConfig
from codur.llm import create_llm, create_llm_profile

console = Console()


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

    # Add nodes
    workflow.add_node("plan", lambda state: plan_node(state, llm, config))
    workflow.add_node("delegate", lambda state: delegate_node(state, config))
    workflow.add_node("tool", lambda state: tool_node(state, config))
    workflow.add_node("execute", lambda state: execute_node(state, config))
    workflow.add_node("review", lambda state: review_node(state, llm, config))

    # Set entry point
    workflow.set_entry_point("plan")

    # Add edges
    workflow.add_conditional_edges(
        "plan",
        should_delegate,
        {
            "delegate": "delegate",
            "tool": "tool",
            "end": END,
        }
    )

    workflow.add_edge("delegate", "execute")
    workflow.add_edge("execute", "review")
    workflow.add_edge("tool", "review")

    workflow.add_conditional_edges(
        "review",
        should_continue,
        {
            "continue": "plan",
            "end": END,
        }
    )

    # Compile the graph with increased recursion limit for trial-error loops
    # Each loop iteration uses multiple nodes (plan→delegate→execute→review→continue)
    # With max_iterations=10, we need at least 10*5=50 recursion depth
    try:
        return workflow.compile(recursion_limit=100)
    except TypeError:
        return workflow.compile()
