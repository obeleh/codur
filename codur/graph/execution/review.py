"""Review node for verifying and routing fix results."""

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.graph.node_types import ReviewNodeResult
from codur.constants import (
    REF_AGENT_CODING,
    ACTION_CONTINUE,
    ACTION_END,
)
from codur.graph.state_operations import (
    is_verbose,
    get_messages,
    prune_messages,
    get_iterations,
    get_agent_outcome,
    get_outcome_result,
    get_tool_calls,
    get_error_hashes,
    was_local_repair_attempted,
)
from codur.tools.project_discovery import get_primary_entry_point
from codur.utils.text_helpers import truncate_lines
from .verification import _verify_fix
from .repair import _attempt_local_repair

console = Console()


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
    if is_verbose(state):
        console.print("[bold magenta]Reviewing result...[/bold magenta]")

    outcome = get_agent_outcome(state)
    result = get_outcome_result(state)
    iterations = get_iterations(state)
    max_iterations = config.runtime.max_iterations

    # If we just ran tools to gather context (e.g., read_file), route to coding agent.
    # BUT: If agent_call was also executed, skip this routing - the result is the implementation.
    if outcome.get("agent") == "tools":
        tool_calls = get_tool_calls(state)
        has_read_file = any(call.get("tool") == "read_file" for call in tool_calls)
        has_agent_call = any(call.get("tool") == "agent_call" for call in tool_calls)

        # Only route to coding agent if we read files but didn't call an agent yet
        if has_read_file and not has_agent_call:
            if is_verbose(state):
                tool_names = [call.get("tool") for call in tool_calls if call.get("tool")]
                tool_label = ", ".join(tool_names) if tool_names else "tool calls"
                console.print(
                    f"[dim]Tool calls completed ({tool_label}; auto-injected follow-ups may have run) - "
                    "delegating to codur-coding[/dim]"
                )

            return {
                "final_response": result,
                "next_action": ACTION_CONTINUE,
            }

    if is_verbose(state):
        console.print(f"[dim]Result status: {outcome.get('status', 'unknown')}[/dim]")
        console.print(f"[dim]Result length: {len(result)} chars[/dim]")
        console.print(f"[dim]Iteration: {iterations}/{max_iterations}[/dim]")

    # Check if this was a tool result (file read, etc) - skip verification for those
    # BUT: If agent_call was executed, it's an implementation result, not just a tool result
    agent_name = outcome.get("agent", "")
    tool_calls = get_tool_calls(state)
    has_agent_call = any(call.get("tool") == "agent_call" for call in tool_calls)

    is_tool_result = (agent_name == "tools" and not has_agent_call) or (isinstance(result, str) and result.startswith("Error") and len(result) < 200)

    # Check if this was a bug fix / debug task by looking at original message
    messages = get_messages(state)
    original_task = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            original_task = msg.content.lower()
            break

    is_fix_task = original_task and any(keyword in original_task for keyword in [
        "fix", "bug", "error", "debug", "issue", "broken", "incorrect", "wrong",
        "implement", "write", "create", "complete", "build"
    ])

    # Try verification if it's a fix task (but not a tool result) and we haven't exceeded iterations
    if is_fix_task and not is_tool_result and iterations < max_iterations - 1:  # Leave room for one more attempt
        verification_result = _verify_fix(state, config)

        if verification_result["success"]:
            if is_verbose(state):
                console.print(f"[green]✓ Verification passed![/green]")
                console.print(f"[dim]{verification_result['message']}[/dim]")
            return {
                "final_response": result,
                "next_action": ACTION_END,
            }
        else:
            # Check for repeated errors (agent is stuck)
            error_msg = verification_result.get("message", "")
            current_error_hash = hash(error_msg)
            error_history = get_error_hashes(state)

            # If same error appears 2+ times in recent history, agent is stuck
            if error_history and current_error_hash in error_history[-3:]:
                if is_verbose(state):
                    console.print(f"[red]✗ Repeated error detected - agent is stuck, stopping[/red]")
                return {
                    "final_response": result,
                    "next_action": ACTION_END,
                }

            # Track this error for future checks
            error_history.append(current_error_hash)

            if not was_local_repair_attempted(state):
                repair_result = _attempt_local_repair(state)
                if repair_result["success"]:
                    if is_verbose(state):
                        console.print(f"[green]✓ Local repair succeeded[/green]")
                        console.print(f"[dim]{repair_result['message']}[/dim]")
                    return {
                        "final_response": repair_result["message"],
                        "next_action": ACTION_END,
                        "local_repair_attempted": True,
                    }
                if is_verbose(state):
                    console.print(f"[yellow]⚠ Local repair failed[/yellow]")
                    console.print(f"[dim]{repair_result['message']}[/dim]")

            # Verification failed - decide whether to retry with coding or replan
            # After 3 failed attempts, route back to planning for a fresh approach
            should_replan = iterations >= 3

            if is_verbose(state):
                if should_replan:
                    console.print(f"[yellow]⚠ Verification failed - routing back to planning for fresh approach[/yellow]")
                else:
                    console.print(f"[yellow]⚠ Verification failed - will retry with coding agent[/yellow]")
                console.print(f"[dim]{verification_result['message'][:200]}[/dim]")

            # Build a structured error message for the agent
            error_parts = ["Verification failed: Output does not match expected."]

            # Determine error type: output mismatch vs execution error
            if "expected_truncated" in verification_result:
                # This was an output mismatch with expected.txt
                error_parts = ["Verification failed: Output does not match expected."]
                error_parts.append(f"\n=== Expected Output ===\n{verification_result['expected_truncated']}")
                if "actual_truncated" in verification_result:
                    error_parts.append(f"\n=== Actual Output ===\n{verification_result['actual_truncated']}")
            else:
                # This was an execution error (exit code != 0)
                if verification_result.get("return_code"):
                    error_parts[0] = f"Verification failed: Code exited with code {verification_result['return_code']}"
                if verification_result.get("stdout"):
                    stdout_content = truncate_lines(verification_result['stdout'], max_lines=15)
                    error_parts.append(f"\n=== Standard Output ===\n{stdout_content}")

            # Include stderr if available (for both error types)
            if verification_result.get("stderr"):
                stderr_content = truncate_lines(verification_result['stderr'], max_lines=15)
                error_parts.append(f"\n=== Error/Exception ===\n{stderr_content}")

            # Try to include current implementation for context
            cwd = Path.cwd()
            entry_point_name = get_primary_entry_point(root=cwd)
            if entry_point_name and not entry_point_name.startswith("Error"):
                entry_point_file = cwd / entry_point_name
                if entry_point_file.exists():
                    try:
                        impl_content = entry_point_file.read_text()
                        if len(impl_content) < 3000:  # Only include if not too large
                            error_parts.append(f"\n=== Current Implementation ({entry_point_name}) ===\n```python\n{impl_content}\n```")
                        else:
                            error_parts.append(f"\n[Current {entry_point_name} is {len(impl_content)} chars - impl too large to display, check what's wrong with current code]")
                    except Exception as read_err:
                        pass

            error_parts.append("\n=== Action ===\nAnalyze the output mismatch and fix the implementation to match expected output.")

            error_message = SystemMessage(content="\n".join(error_parts))

            # Prune old messages to prevent context explosion
            current_messages = get_messages(state)
            pruned_messages = prune_messages(current_messages + [error_message])

            result_dict = {
                "final_response": result,
                "next_action": ACTION_CONTINUE,
                "messages": pruned_messages,
                "local_repair_attempted": True,
                "error_hashes": error_history,
            }

            # After 3 failed attempts, clear selected_agent to route back to planning
            # This allows the planner to try a different approach
            if should_replan:
                result_dict["selected_agent"] = None

            return result_dict

    # For tool results (file reads, etc), continue back to planning with the result as context
    if is_tool_result:
        if is_verbose(state):
            console.print(f"[dim]Tool result: continuing to planning phase with context[/dim]")
        return {
            "final_response": None,
            "next_action": ACTION_CONTINUE,  # Go back to planning to use this context
        }

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
    }
