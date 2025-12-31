"""Review node for verifying and routing fix results."""
import json
from traceback import format_exc

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.graph.node_types import ReviewNodeResult
from codur.constants import (
    ACTION_CONTINUE,
    ACTION_END,
)
from codur.graph.state_operations import (
    is_verbose,
    get_iterations,
    get_agent_outcome,
    get_outcome_result,
    get_error_hashes,
    get_first_human_message_content,
)
from .verification_agent import verification_agent_node

console = Console()

# TODO: llm parameter is currently unused, do we want to pass it to verification_agent_node?
def review_node(state: AgentState, llm: BaseChatModel, config: CodurConfig) -> ReviewNodeResult:
    """Review node: Check if the result is satisfactory.

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

    outcome = get_agent_outcome(state)
    result = get_outcome_result(state)
    iterations = get_iterations(state)
    max_iterations = config.runtime.max_iterations

    if verbose:
        console.print(f"[dim]Result status: {outcome.get('status', 'unknown')}[/dim]")
        console.print(f"[dim]Result length: {len(result)} chars[/dim]")
        console.print(f"[dim]Iteration: {iterations}/{max_iterations}[/dim]")

    # If only tools ran (no agent did actual work), continue to let an agent work
    if outcome.get("agent") == "tools":
        return {
            "final_response": result,
            "next_action": ACTION_CONTINUE,
            "next_step_suggestion": None,
        }

    # Check if this was a bug fix / debug task by looking at original message
    original_task = get_first_human_message_content(state)
    original_task_lower = original_task.lower() if original_task else ""

    is_fix_task = original_task_lower and any(keyword in original_task_lower for keyword in [
        "fix", "bug", "error", "debug", "issue", "broken", "incorrect", "wrong",
        "implement", "write", "create", "complete", "build"
    ])

    # Try verification if it's a fix task and we haven't exceeded iterations
    if is_fix_task and iterations < max_iterations - 1:
        verification_outcome = verification_agent_node(state, config)

        if verification_outcome["agent_outcome"]["status"] == "success":
            if is_verbose(state):
                console.print(f"[green]✓ Verification passed![/green]")
                result = verification_outcome["agent_outcome"]["result"]
                console.print(f"[dim]{verification_outcome['agent_outcome']['result']}[/dim]")
            return {
                "final_response": result,
                "next_action": ACTION_END,
                "next_step_suggestion": None,
            }
        else:
            messages = verification_outcome["messages"]
            last_message = messages[-1] if messages else "No last message?"

            current_error_hash = hash(last_message.content)
            error_history = get_error_hashes(state)

            # If same error appears 2+ times in recent history, agent is stuck
            if error_history and current_error_hash in error_history[-3:]:
                if is_verbose(state):
                    console.print(f"[red]✗ Repeated error detected - agent is stuck, stopping[/red]")
                return {
                    "final_response": result,
                    "next_action": ACTION_END,
                    "next_step_suggestion": None,
                }

            # Track this error for future checks
            error_history.append(current_error_hash)

            # Verification failed - decide whether to retry with coding or replan
            # After 3 failed attempts, route back to planning for a fresh approach
            should_replan = iterations >= 3

            if is_verbose(state):
                if should_replan:
                    console.print(f"[yellow]⚠ Verification failed - routing back to planning for fresh approach[/yellow]")
                else:
                    console.print(f"[yellow]⚠ Verification failed - will retry with coding agent[/yellow]")

                response = _format_verification_response(last_message)
                console.print(f"[dim]{response}[/dim]")

            result_dict = {
                "final_response": result,
                "next_action": ACTION_CONTINUE,
                "messages": messages,
                "error_hashes": error_history,
                "next_step_suggestion": verification_outcome.get("next_step_suggestion"),
            }

            # After 3 failed attempts, clear selected_agent to route back to planning
            # This allows the planner to try a different approach
            if should_replan:
                result_dict["selected_agent"] = None

            return result_dict

    # Accept result if:
    # - Not a fix task
    # - Exceeded max iterations
    # - Verification passed (handled above)
    if is_verbose(state):
        if iterations >= max_iterations - 1:
            console.print(f"[yellow]⚠ Max iterations reached - accepting result[/yellow]")
        else:
            console.print(f"[green]✓ Review complete - accepting result[/green]")

    return {
        "final_response": result,
        "next_action": ACTION_END,
        "next_step_suggestion": None,
    }


def _format_verification_response(last_message: BaseMessage) -> str:
    response = last_message.content
    try:
        loaded = json.loads(response)
        output = loaded["output"]
        response = f"passed: {output['passed']}"
        if "error" in loaded:
            response += f"\nError: {loaded['error']}"
        response += f"\n{output['expected']} vs\n{output['actual']}"
    except Exception as ex:
        print(f"!!!!!!!! {format_exc(ex)}")
    return response
