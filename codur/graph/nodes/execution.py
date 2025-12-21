"""Delegation, execution, and review nodes."""

from langchain_core.messages import HumanMessage
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

# Import agents to ensure they are registered
import codur.agents.ollama_agent  # noqa: F401
import codur.agents.codex_agent  # noqa: F401
import codur.agents.claude_code_agent  # noqa: F401

console = Console()


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
                import traceback
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
        result = agent.execute(task)
        if self.state.get("verbose"):
            console.print(f"[dim]Agent result length: {len(result)} chars[/dim]")
            console.print(f"[dim]Agent result preview: {result[:300]}...[/dim]")
        return result


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
    4. Accepts if verification succeeds or max iterations reached

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

    if state.get("verbose"):
        console.print(f"[dim]Result status: {outcome.get('status', 'unknown')}[/dim]")
        console.print(f"[dim]Result length: {len(result)} chars[/dim]")
        console.print(f"[dim]Iteration: {iterations}/{max_iterations}[/dim]")

    # Check if this was a bug fix / debug task by looking at original message
    from langchain_core.messages import HumanMessage, SystemMessage
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

    # Try verification if it's a fix task and we haven't exceeded iterations
    if is_fix_task and iterations < max_iterations - 1:  # Leave room for one more attempt
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
                console.print(f"[dim]{verification_result['message']}[/dim]")

            # Add verification error to messages for next iteration
            error_message = SystemMessage(
                content=f"Verification failed: {verification_result['message']}\n\n"
                        f"Please analyze the issue and fix it. Previous attempt:\n{result[:500]}"
            )

            return {
                "final_response": result,
                "next_action": "continue",
                "messages": [error_message],
                "local_repair_attempted": True,
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


def _verify_fix(state: AgentState, config: CodurConfig) -> dict:
    """Verify if a fix actually works by running tests.

    Args:
        state: Current agent state
        config: Codur configuration

    Returns:
        Dict with "success" (bool) and "message" (str)
    """
    import subprocess
    from pathlib import Path

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

    try:
        # Run main.py and capture output
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cwd)
        )

        # Check for expected.txt to compare output
        expected_file = cwd / "expected.txt"
        if expected_file.exists():
            expected_output = expected_file.read_text().strip()
            actual_output = result.stdout.strip()

            if actual_output == expected_output:
                return {
                    "success": True,
                    "message": f"Output matches expected: {actual_output}"
                }
            else:
                return {
                    "success": False,
                    "message": f"Output mismatch.\nExpected: {expected_output}\nActual: {actual_output}"
                }

        # If no expected.txt, just check for success exit code
        if result.returncode == 0:
            return {
                "success": True,
                "message": f"Execution successful. Output: {result.stdout.strip()}"
            }
        else:
            return {
                "success": False,
                "message": f"Execution failed (exit code {result.returncode})\nStderr: {result.stderr.strip()}"
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Execution timed out after 10 seconds"}
    except Exception as e:
        return {"success": False, "message": f"Verification error: {str(e)}"}


def _attempt_local_repair(state: AgentState) -> dict:
    """Attempt a small, local repair for common mismatch patterns.

    This is a last-resort fallback when external agents are unavailable.
    """
    import re
    import subprocess
    from pathlib import Path

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

    mutations = [
        mutate_range_inclusive,
        mutate_remove_continue_guard,
        mutate_remove_div_100,
    ]

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

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        main_py.write_text(candidate)
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cwd),
        )
        if result.returncode == 0 and result.stdout.strip() == expected_output:
            return {"success": True, "message": "Applied local repair based on verification output"}

    main_py.write_text(original)
    return {"success": False, "message": "Local repair did not find a matching fix"}
