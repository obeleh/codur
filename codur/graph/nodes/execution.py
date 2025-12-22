"""Delegation, execution, and review nodes."""

import re
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.llm import create_llm_profile, create_llm
from codur.agents import AgentRegistry
from codur.graph.nodes.types import DelegateNodeResult, ExecuteNodeResult, ReviewNodeResult
from codur.graph.nodes.utils import (
    _resolve_agent_profile,
    _resolve_agent_reference,
    _normalize_messages,
)
from codur.graph.nodes.tool_detection import create_default_tool_detector
from codur.tools import read_file, write_file

# Import agents to ensure they are registered
import codur.agents.ollama_agent  # noqa: F401
import codur.agents.codex_agent  # noqa: F401
import codur.agents.claude_code_agent  # noqa: F401

console = Console()
_TOOL_DETECTOR = create_default_tool_detector()


class AgentExecutor:
    def __init__(self, state: AgentState, config: CodurConfig) -> None:
        self.state = state
        self.config = config
        self.default_agent = config.agents.preferences.default_agent or "agent:ollama"
        self.agent_name = state["agent_outcome"].get("agent", self.default_agent)
        self.agent_name, self.profile_override = _resolve_agent_profile(config, self.agent_name)
        self.resolved_agent = _resolve_agent_reference(self.agent_name)

    def execute(self) -> ExecuteNodeResult:
        if self.state.get("verbose"):
            console.print(f"[bold green]Executing with {self.agent_name}...[/bold green]")
            console.print(f"[dim]Resolved agent: {self.resolved_agent}[/dim]")

        messages = _normalize_messages(self.state.get("messages"))
        last_message = messages[-1] if messages else None

        if not last_message:
            return {"agent_outcome": {"agent": self.agent_name, "result": "No task provided", "status": "error"}}

        if self.state.get("verbose"):
            task_preview = str(last_message.content if hasattr(last_message, "content") else last_message)[:200]
            console.print(f"[dim]Task: {task_preview}...[/dim]")

        task = last_message.content if hasattr(last_message, "content") else str(last_message)

        try:
            if self.agent_name.startswith("llm:"):
                result = self._execute_llm_profile(task)
            else:
                result = self._execute_agent(task)

            if self.state.get("verbose"):
                console.print("[green]✓ Execution completed successfully[/green]")

            return {
                "agent_outcome": {
                    "agent": self.agent_name,
                    "result": result,
                    "status": "success",
                }
            }
        except Exception as exc:
            console.print(f"[red]✗ Error executing {self.agent_name}: {str(exc)}[/red]")
            if self.state.get("verbose"):
                console.print(f"[red]{traceback.format_exc()}[/red]")
            return {
                "agent_outcome": {
                    "agent": self.agent_name,
                    "result": str(exc),
                    "status": "error",
                }
            }

    def _execute_llm_profile(self, task: str) -> str:
        profile_name = self.agent_name.split(":", 1)[1]
        if self.state.get("verbose"):
            console.print(f"[dim]Using LLM profile: {profile_name}[/dim]")
        llm = create_llm_profile(self.config, profile_name)
        response = llm.invoke([HumanMessage(content=task)])
        result = response.content
        if self.state.get("verbose"):
            console.print(f"[dim]LLM response length: {len(result)} chars[/dim]")
            console.print(f"[dim]LLM response preview: {result[:300]}...[/dim]")
        return result

    def _execute_agent(self, task: str) -> str:
        agent_config = self.config.agents.configs.get(self.resolved_agent)
        if agent_config and getattr(agent_config, "type", None) == "llm":
            return self._execute_llm_agent(agent_config, task)
        return self._execute_registered_agent(task)

    def _execute_llm_agent(self, agent_config, task: str) -> str:
        model = agent_config.config.get("model")
        matching_profile = None
        for profile_name, profile in self.config.llm.profiles.items():
            if profile.model == model:
                matching_profile = profile_name
                break

        llm = create_llm_profile(self.config, matching_profile) if matching_profile else create_llm(self.config)
        response = llm.invoke([HumanMessage(content=task)])
        result = response.content
        if self.state.get("verbose"):
            console.print(f"[dim]LLM response length: {len(result)} chars[/dim]")
            console.print(f"[dim]LLM response preview: {result[:300]}...[/dim]")
        return result

    def _execute_registered_agent(self, task: str) -> str:
        agent_class = AgentRegistry.get(self.resolved_agent)
        if self.state.get("verbose"):
            console.print(f"[dim]Using agent class: {agent_class.__name__ if agent_class else 'None'}[/dim]")
        if not agent_class:
            available = ", ".join(AgentRegistry.list_agents())
            raise ValueError(f"Unknown agent: {self.resolved_agent}. Available agents: {available}")

        agent = agent_class(self.config, override_config=self.profile_override)

        # Wrap agent in tool-using loop for iterative execution
        return self._execute_agent_with_tools(agent, task)

    def _execute_agent_with_tools(self, agent, task: str, max_tool_iterations: int = 5) -> str:
        """Execute agent with automatic tool call detection and execution.

        This implements a tool-using loop where:
        1. Agent generates response
        2. Response is checked for tool calls
        3. Tools are executed and results fed back
        4. Loop continues until no more tool calls or max iterations
        """
        messages = [HumanMessage(content=task)]
        result = None
        tool_iteration = 0

        while tool_iteration < max_tool_iterations:
            # Get agent response
            if tool_iteration == 0:
                # First call: use the task directly
                result = agent.execute(task)
            else:
                # Subsequent calls: construct message with tool results
                full_prompt = self._build_prompt_with_tool_results(task, messages)
                result = agent.execute(full_prompt)

            if self.state.get("verbose"):
                console.print(f"[dim]Agent iteration {tool_iteration}: result length {len(result)} chars[/dim]")

            # Check for tool calls in the response
            tool_calls = _TOOL_DETECTOR.detect(result)

            if not tool_calls:
                # No tool calls detected, return final result
                if self.state.get("verbose"):
                    console.print(f"[dim]Agent finished after {tool_iteration} iterations[/dim]")
                return result

            # Execute tools and collect results
            if self.state.get("verbose"):
                console.print(f"[yellow]Detected {len(tool_calls)} tool call(s), executing...[/yellow]")

            tool_results = self._execute_tool_calls(tool_calls)
            messages.append(AIMessage(content=result))
            messages.append(SystemMessage(content=f"Tool results:\n{tool_results}"))

            tool_iteration += 1

        if self.state.get("verbose"):
            console.print(f"[yellow]Max tool iterations ({max_tool_iterations}) reached[/yellow]")
        return result

    def _build_prompt_with_tool_results(self, original_task: str, messages: list) -> str:
        """Build a prompt that includes tool results for the agent to continue."""
        # Extract tool results from messages
        tool_results = []
        for msg in messages:
            if isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:"):
                tool_results.append(msg.content)

        if tool_results:
            return f"{original_task}\n\n{chr(10).join(tool_results)}\n\nContinue fixing the implementation based on the tool results."
        return original_task

    def _execute_tool_calls(self, tool_calls: list[dict]) -> str:
        """Execute detected tool calls and return results."""
        results = []
        root = Path.cwd()
        allow_outside_root = self.config.runtime.allow_outside_workspace

        for call in tool_calls:
            tool_name = call.get("tool")
            args = call.get("args", {})

            if not isinstance(args, dict):
                args = {}

            try:
                if tool_name == "read_file":
                    output = read_file(root=root, allow_outside_root=allow_outside_root, **args)
                    results.append(f"read_file: {args.get('path', 'unknown')} -> {len(output) if isinstance(output, str) else len(str(output))} chars")
                elif tool_name == "write_file":
                    output = write_file(root=root, allow_outside_root=allow_outside_root, **args)
                    results.append(f"write_file: {args.get('path', 'unknown')} -> {output}")
                else:
                    results.append(f"Unknown tool: {tool_name}")
            except Exception as e:
                results.append(f"{tool_name} failed: {str(e)}")

        return "\n".join(results) if results else "Tools executed (no output)"


def delegate_node(state: AgentState, config: CodurConfig) -> DelegateNodeResult:
    """Delegation node: Route to the appropriate agent.

    Args:
        state: Current agent state containing selected_agent
        config: Codur configuration

    Returns:
        Dictionary with agent_outcome containing agent name and status
    """
    if state.get("verbose"):
        console.print("[bold cyan]Delegating task...[/bold cyan]")

    # Use the agent selected by the plan_node, fallback to configured default
    default_agent = config.agents.preferences.default_agent or "agent:ollama"
    selected_agent = state.get("selected_agent", default_agent)

    return {
        "agent_outcome": {
            "agent": selected_agent,
            "status": "delegated"
        }
    }


def execute_node(state: AgentState, config: CodurConfig) -> ExecuteNodeResult:
    """Execution node: Actually run the delegated task.

    Args:
        state: Current agent state with agent_outcome and messages
        config: Codur configuration for agent setup

    Returns:
        Dictionary with agent_outcome containing execution result
    """
    return AgentExecutor(state, config).execute()


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
    if state.get("verbose"):
        console.print("[bold magenta]Reviewing result...[/bold magenta]")

    outcome = state.get("agent_outcome", {})
    result = outcome.get("result", "")
    iterations = state.get("iterations", 0)
    max_iterations = config.runtime.max_iterations

    # If we just ran tools to gather context (e.g., read_file), route to coding agent.
    # BUT: If agent_call was also executed, skip this routing - the result is the implementation.
    if outcome.get("agent") == "tools":
        tool_calls = state.get("tool_calls", []) or []
        has_read_file = any(call.get("tool") == "read_file" for call in tool_calls)
        has_agent_call = any(call.get("tool") == "agent_call" for call in tool_calls)

        # Only route to coding agent if we read files but didn't call an agent yet
        if has_read_file and not has_agent_call:
            if state.get("verbose"):
                console.print("[dim]Tool read_file completed - delegating to codur-coding[/dim]")
            return {
                "final_response": result,
                "next_action": "continue",
                "selected_agent": "agent:codur-coding",
            }

    if state.get("verbose"):
        console.print(f"[dim]Result status: {outcome.get('status', 'unknown')}[/dim]")
        console.print(f"[dim]Result length: {len(result)} chars[/dim]")
        console.print(f"[dim]Iteration: {iterations}/{max_iterations}[/dim]")

    # Check if this was a tool result (file read, etc) - skip verification for those
    # BUT: If agent_call was executed, it's an implementation result, not just a tool result
    agent_name = outcome.get("agent", "")
    tool_calls = state.get("tool_calls", []) or []
    has_agent_call = any(call.get("tool") == "agent_call" for call in tool_calls)

    is_tool_result = (agent_name == "tools" and not has_agent_call) or (isinstance(result, str) and result.startswith("Error") and len(result) < 200)

    # Check if this was a bug fix / debug task by looking at original message
    messages = state.get("messages", [])
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
            if state.get("verbose"):
                console.print(f"[green]✓ Verification passed![/green]")
                console.print(f"[dim]{verification_result['message']}[/dim]")
            return {
                "final_response": result,
                "next_action": "end",
            }
        else:
            # Check for repeated errors (agent is stuck)
            error_msg = verification_result.get("message", "")
            current_error_hash = hash(error_msg)
            error_history = state.get("error_hashes", [])

            # If same error appears 2+ times in recent history, agent is stuck
            if error_history and current_error_hash in error_history[-3:]:
                if state.get("verbose"):
                    console.print(f"[red]✗ Repeated error detected - agent is stuck, stopping[/red]")
                return {
                    "final_response": result,
                    "next_action": "end",
                }

            # Track this error for future checks
            error_history.append(current_error_hash)

            local_repair_attempted = state.get("local_repair_attempted", False)
            if not local_repair_attempted:
                repair_result = _attempt_local_repair(state)
                if repair_result["success"]:
                    if state.get("verbose"):
                        console.print(f"[green]✓ Local repair succeeded[/green]")
                        console.print(f"[dim]{repair_result['message']}[/dim]")
                    return {
                        "final_response": repair_result["message"],
                        "next_action": "end",
                        "local_repair_attempted": True,
                    }
                if state.get("verbose"):
                    console.print(f"[yellow]⚠ Local repair failed[/yellow]")
                    console.print(f"[dim]{repair_result['message']}[/dim]")

            # Verification failed - loop back with error message
            if state.get("verbose"):
                console.print(f"[yellow]⚠ Verification failed - will retry[/yellow]")
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
                    stdout_content = _truncate_output(verification_result['stdout'], max_lines=15)
                    error_parts.append(f"\n=== Standard Output ===\n{stdout_content}")

            # Include stderr if available (for both error types)
            if verification_result.get("stderr"):
                stderr_content = _truncate_output(verification_result['stderr'], max_lines=15)
                error_parts.append(f"\n=== Error/Exception ===\n{stderr_content}")

            # Try to include current main.py for context
            cwd = Path.cwd()
            main_py = cwd / "main.py"
            if main_py.exists():
                try:
                    main_content = main_py.read_text()
                    if len(main_content) < 3000:  # Only include if not too large
                        error_parts.append(f"\n=== Current Implementation (main.py) ===\n```python\n{main_content}\n```")
                    else:
                        error_parts.append(f"\n[Current main.py is {len(main_content)} chars - impl too large to display, check what's wrong with current code]")
                except Exception as read_err:
                    pass

            error_parts.append("\n=== Action ===\nAnalyze the output mismatch and fix the implementation to match expected output.")

            error_message = SystemMessage(content="\n".join(error_parts))

            # Prune old messages to prevent context explosion
            current_messages = state.get("messages", [])
            pruned_messages = _prune_messages(current_messages + [error_message])

            return {
                "final_response": result,
                "next_action": "continue",
                "messages": pruned_messages,
                "local_repair_attempted": True,
                "error_hashes": error_history,
            }

    # For tool results (file reads, etc), continue back to planning with the result as context
    if is_tool_result:
        if state.get("verbose"):
            console.print(f"[dim]Tool result: continuing to planning phase with context[/dim]")
        return {
            "final_response": None,
            "next_action": "continue",  # Go back to planning to use this context
        }

    # Accept result if:
    # - Not a fix task
    # - Exceeded max iterations
    # - Verification passed (handled above)
    if state.get("verbose"):
        if iterations >= max_iterations - 1:
            console.print(f"[yellow]⚠ Max iterations reached - accepting result[/yellow]")
        else:
            console.print(f"[green]✓ Review complete - accepting result[/green]")

    return {
        "final_response": result,
        "next_action": "end",
    }


def _truncate_output(text: str, max_lines: int = 30) -> str:
    """Truncate output to a reasonable length for display and agent processing.

    Using 30 lines (increased from 20) to give agents more context to understand
    what went wrong and fix issues. This improves reliability.
    """
    lines = text.split('\n')
    if len(lines) > max_lines:
        truncated = '\n'.join(lines[:max_lines])
        return f"{truncated}\n... ({len(lines) - max_lines} more lines)"
    return text


def _prune_messages(messages: list, max_to_keep: int = 10) -> list:
    """Prune old messages to prevent context explosion while preserving learning context.

    Keeps:
    - Original HumanMessage(s) - typically the first message
    - Recent AIMessage(s) - agent's recent attempts (so it learns from them)
    - Recent SystemMessage(s) - recent error/verification messages for context

    This helps agents learn from their mistakes by seeing their own attempts and the feedback.
    """
    if len(messages) <= max_to_keep:
        return messages

    # Find the first human message (original task)
    first_human_idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            first_human_idx = i
            break

    if first_human_idx is None:
        # No human message, just keep last N
        return messages[-max_to_keep:]

    # Keep original task
    pruned = messages[:first_human_idx + 1]

    # Keep recent agent attempts and error messages together (last 5 attempts)
    # This way agent sees: "I tried X, got error Y, I tried Z, got error W"
    recent_count = 0
    max_recent = 5
    for i, msg in enumerate(reversed(messages[first_human_idx + 1:])):
        # Keep both AIMessage (agent attempts) and SystemMessage (errors) from recent history
        if isinstance(msg, (AIMessage, SystemMessage)):
            if isinstance(msg, SystemMessage) and "Verification failed" not in msg.content:
                continue  # Skip non-error system messages
            pruned.append(msg)
            recent_count += 1
            if recent_count >= max_recent:
                break

    # Reverse to maintain chronological order
    if len(pruned) > first_human_idx + 1:
        recent = pruned[first_human_idx + 1:]
        recent.reverse()
        pruned = pruned[:first_human_idx + 1] + recent

    return pruned


def _verify_fix(state: AgentState, config: CodurConfig) -> dict:
    """Verify if a fix actually works by running tests.

    Implements streaming verification with early exit on output mismatch.
    This prevents wasting time on obviously failed implementations.

    Args:
        state: Current agent state
        config: Codur configuration

    Returns:
        Dict with "success" (bool), "message" (str), and other diagnostic info
    """
    verbose = state.get("verbose", False)

    # Look for main.py in current directory
    cwd = Path.cwd()
    main_py = cwd / "main.py"

    if not main_py.exists():
        if verbose:
            console.print("[dim]No main.py found - skipping verification[/dim]")
        return {"success": True, "message": "No verification file found"}

    if verbose:
        console.print(f"[dim]Running verification: python main.py[/dim]")

    # Check for expected.txt to enable streaming verification
    expected_file = cwd / "expected.txt"
    use_streaming = expected_file.exists()

    if use_streaming:
        expected_output = expected_file.read_text().strip()
        expected_lines = expected_output.split('\n')

        try:
            # Run with streaming for early exit on mismatch
            process = subprocess.Popen(
                ["python", "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd)
            )

            output_lines = []
            mismatch_at_line = None

            try:
                # Stream output line by line and compare
                for line in process.stdout:
                    output_lines.append(line.rstrip('\n'))
                    line_idx = len(output_lines) - 1

                    # Check for mismatch (early exit)
                    if line_idx < len(expected_lines):
                        if line.rstrip('\n') != expected_lines[line_idx]:
                            mismatch_at_line = line_idx
                            process.terminate()
                            break
                    else:
                        # Output has more lines than expected
                        mismatch_at_line = line_idx
                        process.terminate()
                        break

                # Wait for process to finish
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

                stderr = process.stderr.read() if process.stderr else ""

            except Exception as e:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                return {"success": False, "message": f"Verification interrupted: {str(e)}"}

            # Check results
            actual_output = '\n'.join(output_lines).strip()

            if mismatch_at_line is None and actual_output == expected_output:
                return {
                    "success": True,
                    "message": f"Output matches expected: {actual_output}"
                }
            else:
                # Build detailed mismatch info
                return {
                    "success": False,
                    "message": f"Output mismatch.\nActual: {actual_output}\nExpected: {expected_output}",
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "expected_truncated": _truncate_output(expected_output),
                    "actual_truncated": _truncate_output(actual_output),
                    "mismatch_at_line": mismatch_at_line,
                    "stderr": stderr.strip() if stderr else None
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Execution timed out after 10 seconds"}
        except Exception as e:
            return {"success": False, "message": f"Verification error: {str(e)}"}
    else:
        # No expected.txt - use standard verification
        try:
            result = subprocess.run(
                ["python", "main.py"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(cwd)
            )

            # Just check for success exit code
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Execution successful. Output: {result.stdout.strip()}"
                }
            else:
                std_out_stripped = result.stdout.strip()
                std_err_stripped = result.stderr.strip()
                if verbose:
                    # print std out and stderr for debugging
                    console.print(f"[dim]Stdout: {std_out_stripped}[/dim]")
                    console.print(f"[dim]Stderr: {std_err_stripped}[/dim]")
                return {
                    "success": False,
                    "message": f"Execution failed (exit code {result.returncode})\nStderr: {std_err_stripped}",
                    "stderr": std_err_stripped,
                    "stdout": std_out_stripped,
                    "return_code": result.returncode
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Execution timed out after 10 seconds"}
        except Exception as e:
            return {"success": False, "message": f"Verification error: {str(e)}"}


def _attempt_local_repair(state: AgentState) -> dict:
    """Attempt a small, local repair for common mismatch patterns.

    This is a last-resort fallback when external agents are unavailable.
    Uses parallel execution for faster mutation testing.
    """
    cwd = Path.cwd()
    main_py = cwd / "main.py"
    expected_file = cwd / "expected.txt"

    if not main_py.exists() or not expected_file.exists():
        return {"success": False, "message": "No repair target found"}

    original = main_py.read_text()
    expected_output = expected_file.read_text().strip()

    def mutate_range_inclusive(text: str) -> str:
        def _replace(match: re.Match) -> str:
            start = match.group(1).strip()
            end = match.group(2).strip()
            if end.endswith("+ 1") or end.endswith("+1"):
                return match.group(0)
            return f"range({start}, {end} + 1)"
        return re.sub(r"\brange\(([^,]+),\s*([^)]+)\)", _replace, text)

    def mutate_remove_continue_guard(text: str) -> str:
        pattern = re.compile(r"^(?P<indent>\s*)if\s+(?P<cond>.+):\n(?P=indent)\s+continue\b", re.MULTILINE)
        return pattern.sub(lambda m: f"{m.group('indent')}if {m.group('cond')}:\n{m.group('indent')}    pass", text)

    def mutate_remove_div_100(text: str) -> str:
        text = re.sub(r"\(([^()]+?)\s*/\s*100(?:\.0+)?\)", r"\1", text)
        return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*/\s*100(?:\.0+)?\b", r"\1", text)

    def mutate_fix_comparison(text: str) -> str:
        """Fix common comparison mistakes like >= vs >, <= vs <"""
        # Try flipping >= to >
        alt1 = text.replace(">=", "__GTE__").replace(">", ">=").replace("__GTE__", ">")
        if alt1 != text:
            return alt1
        # Try flipping <= to <
        alt2 = text.replace("<=", "__LTE__").replace("<", "<=").replace("__LTE__", "<")
        if alt2 != text:
            return alt2
        return text

    def mutate_fix_loop_condition(text: str) -> str:
        """Fix loop conditions that are too strict or too loose"""
        # Change 'while n' to 'while n > 0' (common mistake)
        text = re.sub(r"\bwhile\s+([a-z_]\w*)\s*:", r"while \1 > 0:", text)
        return text

    mutations = [
        mutate_range_inclusive,
        mutate_remove_continue_guard,
        mutate_remove_div_100,
        mutate_fix_comparison,
        mutate_fix_loop_condition,
    ]

    # Build all candidate mutations
    candidates = []
    for mutate in mutations:
        updated = mutate(original)
        if updated != original:
            candidates.append(updated)

    for i in range(len(mutations)):
        for j in range(i + 1, len(mutations)):
            first = mutations[i](original)
            if first == original:
                continue
            second = mutations[j](first)
            if second != first and second != original:
                candidates.append(second)

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    if not unique_candidates:
        return {"success": False, "message": "No applicable mutations found"}

    def try_mutation(candidate_code: str) -> dict:
        """Test a mutation in a temporary file to allow parallel execution."""
        try:
            # Create temp file with mutation
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                dir=str(cwd),
                delete=False
            ) as tmp:
                tmp.write(candidate_code)
                tmp_path = Path(tmp.name)

            try:
                result = subprocess.run(
                    ["python", str(tmp_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(cwd),
                )
                if result.returncode == 0 and result.stdout.strip() == expected_output:
                    return {"success": True, "code": candidate_code}
                return {"success": False}
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            return {"success": False}

    # Run mutations in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(try_mutation, c): c for c in unique_candidates}

        for future in as_completed(futures, timeout=10):
            try:
                result = future.result()
                if result.get("success"):
                    # Found a working mutation - apply it to main.py
                    main_py.write_text(result["code"])
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    return {"success": True, "message": "Applied local repair based on verification output"}
            except Exception:
                continue

    return {"success": False, "message": "Local repair did not find a matching fix"}
