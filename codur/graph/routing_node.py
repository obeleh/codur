"""Review node for verifying and routing fix results."""
import json
from traceback import format_exc

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.graph.node_types import ReviewNodeResult
# Node names for routing
NODE_END = "end"
NODE_VERIFICATION = "verification"
NODE_LLM_PLAN = "llm_plan"
from codur.graph.state_operations import (
    is_verbose,
    get_iterations,
    get_outcome_result,
    ToolOutput, get_messages, get_last_tool_output_from_messages,
    is_coding_agent_session, get_latest_agent_outcome, get_last_tool_output,
    get_selected_agent,
)
from codur.graph.verification_agent import verification_agent_node

console = Console()

def handle_verification_tool_output(messages: list[BaseMessage]) -> ReviewNodeResult | None:
    last_tool_output = get_last_tool_output_from_messages(messages)
    if last_tool_output:
        assert isinstance(last_tool_output, ToolOutput)
        if last_tool_output.tool == "build_verification_response":
            # Access output (return value), not args (input parameters)
            output = last_tool_output.output
            if isinstance(output, dict):
                passed = output.get("passed", False)
                reasoning = output.get("reasoning", "")
            else:
                # Fallback: check args for backward compatibility
                passed = last_tool_output.args.get("passed", False)
                reasoning = last_tool_output.args.get("reasoning", "")

            if passed:
                return ReviewNodeResult(
                    final_response=reasoning,
                    next_action=NODE_END,
                )
            else:
                return ReviewNodeResult(
                    final_response=reasoning,
                    next_action=NODE_LLM_PLAN,
                    next_step_suggestion=None,
                )


def routing_node(state: AgentState, llm: BaseChatModel, config: CodurConfig) -> ReviewNodeResult:
    """Routing node: Review execution results and route to next node.

    Implements trial-error loop for bug fixes:
    1. Detects if this was a fix/debug task
    2. Runs verification (e.g., python main.py)
    3. Loops back if verification fails (up to max_iterations)
    4. Exits early if same error repeats (agent is stuck)
    5. Accepts if verification succeeds or max iterations reached

    Args:
        state: Current agent state with agent_outcome
        llm: Language model (unused but kept for compatibility)
        config: Codur configuration

    Returns:
        Dictionary with final_response and next_action ("end" or "continue")
    """
    verbose = is_verbose(state)
    if verbose:
        console.print("[bold magenta]Reviewing result...[/bold magenta]")

    outcome = get_latest_agent_outcome(state)
    result = get_outcome_result(state)
    iterations = get_iterations(state)
    max_iterations = config.runtime.max_iterations

    if verbose:
        console.print(f"[dim]Result status: {outcome.get('status', 'unknown')}[/dim]")
        console.print(f"[dim]Result length: {len(result)} chars[/dim]")
        console.print(f"[dim]Iteration: {iterations}/{max_iterations}[/dim]")

    if iterations > max_iterations:
        console.print(f"[yellow]⚠ Exceeded max iterations ({max_iterations}) - accepting result[/yellow]")
        return {
            "final_response": result,
            "next_action": NODE_END,
        }

    last_tool_call = get_last_tool_output(state)
    last_tool_name = last_tool_call.tool if last_tool_call else None
    if last_tool_name == "done":
        if verbose:
            console.print(f"[green]✓ 'done' tool detected - accepting result[/green]")
        return ReviewNodeResult(
            final_response=result,
            next_action=NODE_END,
        )
    elif last_tool_name == "build_verification_response":
        # Access output (return value), not args (input parameters)
        output = last_tool_call.output
        if isinstance(output, dict):
            passed = output.get("passed", False)
            reasoning = output.get("reasoning", "")
        else:
            # Fallback: check args for backward compatibility
            passed = last_tool_call.args.get("passed", False)
            reasoning = last_tool_call.args.get("reasoning", "")

        if passed:
            if verbose:
                console.print(f"[green]✓ Verification passed - accepting result[/green]")
            return ReviewNodeResult(
                final_response=reasoning,
                next_action=NODE_END,
            )
        else:
            if verbose:
                console.print(f"[yellow]⚠ Verification failed - continuing to fix[/yellow]")
            return ReviewNodeResult(
                next_action=NODE_VERIFICATION,
            )


    # If only tools ran (no agent did actual work), route to selected agent or back to planning
    last_agent = outcome.get("agent")
    if last_agent == "tools":
        # Check if planning already selected an agent (e.g., action="tool" with agent="codur-coding")
        selected_agent = get_selected_agent(state)
        if selected_agent:
            # Import here to avoid circular dependency
            from codur.graph.main_graph import get_agent_route
            agent_route = get_agent_route(selected_agent)
            if agent_route:
                # Route to the selected agent (e.g., "coding" or "explaining")
                if verbose:
                    console.print(f"[cyan]→ Routing to {agent_route} agent after tools[/cyan]")
                return ReviewNodeResult(
                    next_action=agent_route,
                )
        # No agent selected or not a specialized agent, continue to planning
        return ReviewNodeResult(
            next_action=NODE_LLM_PLAN,
        )
    elif last_agent == "coding":
        return ReviewNodeResult(
            next_action=NODE_VERIFICATION,
        )
    elif last_agent == "review":
        raise ValueError("Didn't expect to land here, this should already be handled above")


    messages = get_messages(state)
    result = handle_verification_tool_output(messages)
    if result:
        return result

    # Try verification if it's a fix task and we haven't exceeded iterations
    if is_coding_agent_session(state) and iterations < max_iterations - 1:
        verification_outcome = verification_agent_node(state, config)
        result = handle_verification_tool_output(verification_outcome["messages"])
        if result:
            return result

    # Accept result if:
    # - Not a fix task
    # - Exceeded max iterations
    # - Verification passed (handled above)
    if is_verbose(state):
        if iterations >= max_iterations - 1:
            console.print(f"[yellow]⚠ Max iterations reached - accepting result[/yellow]")
        else:
            console.print(f"[green]✓ Review complete - accepting result[/green]")

    return ReviewNodeResult(
        next_action=NODE_END,
    )


def _format_verification_response(last_message: BaseMessage) -> str:
    response = last_message.content
    try:
        loaded = json.loads(response)
        if "output" in loaded:
            output = loaded["output"]
            response += f"passed: {output['passed']}"
            response += f"\n{output['expected']} vs\n{output['actual']}"
        if "error" in loaded:
            response += f"\nError: {loaded['error']}"
    except Exception:
        print(f"_format_verification_response error: {format_exc()}")
    return response
