"""Review node for verifying and routing fix results."""
import json
from traceback import format_exc

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
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
    get_first_human_message_content, get_last_tool_output, ToolOutput, get_messages, get_last_tool_output_from_messages,
    is_coding_agent_session,
)
from codur.graph.verification_agent import verification_agent_node

console = Console()

def handle_verification_tool_output(messages: list[BaseMessage]) -> ReviewNodeResult | None:
    last_tool_output = get_last_tool_output_from_messages(messages)
    if last_tool_output:
        assert isinstance(last_tool_output, ToolOutput)
        if last_tool_output.tool == "build_verification_response":

            if last_tool_output.args["passed"]:
                return {
                    "final_response": last_tool_output.args["reasoning"],
                    "next_action": ACTION_END,
                    "next_step_suggestion": None,
                }
            else:
                return {
                    "final_response": last_tool_output.args["reasoning"],
                    "next_action": ACTION_CONTINUE,
                    "next_step_suggestion": None,
                }


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

    return {
        "final_response": result,
        "next_action": ACTION_END,
        "next_step_suggestion": None,
    }


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
