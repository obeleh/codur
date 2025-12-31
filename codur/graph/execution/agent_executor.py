"""Agent executor with tool loop support."""

import traceback
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.utils.llm_helpers import create_and_invoke
from codur.agents import AgentRegistry
from codur.graph.node_types import ExecuteNodeResult
from codur.graph.utils import resolve_agent_profile, resolve_agent_reference
from codur.utils.config_helpers import get_default_agent
from codur.utils.validation import require_config
from codur.graph.state_operations import (
    is_verbose,
    get_messages,
    get_llm_calls,
)
from codur.graph.tool_executor import execute_tool_calls
from codur.utils.llm_calls import LLMCallLimitExceeded

# Import agents to ensure they are registered
import codur.agents.ollama_agent  # noqa: F401
import codur.agents.codex_agent  # noqa: F401
import codur.agents.claude_code_agent  # noqa: F401

console = Console()


class AgentExecutor:
    def __init__(self, state: AgentState, config: CodurConfig, agent_name: Optional[str]=None) -> None:
        self.state = state
        self.config = config
        self.default_agent = get_default_agent(config)
        require_config(
            self.default_agent,
            "agents.preferences.default_agent",
            "agents.preferences.default_agent must be configured",
        )
        if agent_name:
            self.agent_name = agent_name
        else:
            self.agent_name = state["agent_outcome"].get("agent", self.default_agent)
            self.agent_name, self.profile_override = resolve_agent_profile(config, self.agent_name)
        self.resolved_agent = resolve_agent_reference(self.agent_name)

    def execute(self) -> ExecuteNodeResult:
        if is_verbose(self.state):
            console.print(f"[bold green]Executing with {self.agent_name}...[/bold green]")
            console.print(f"[dim]Resolved agent: {self.resolved_agent}[/dim]")

        messages = get_messages(self.state)
        last_message = messages[-1] if messages else None

        if not last_message:
            return {"agent_outcome": {"agent": self.agent_name, "result": "No task provided", "status": "error"}, "next_step_suggestion": None}

        if is_verbose(self.state):
            task_preview = str(last_message.content if hasattr(last_message, "content") else last_message)[:200]
            console.print(f"[dim]Task: {task_preview}...[/dim]")

        task = last_message.content if hasattr(last_message, "content") else str(last_message)

        try:
            if self.agent_name.startswith("llm:"):
                messages_out, result = self._execute_llm_profile(task)
            else:
                messages_out, result = self._execute_agent(task)

            assert isinstance(messages_out, list)
            if is_verbose(self.state):
                console.print("[green]✓ Execution completed successfully[/green]")

            dct = {
                "agent_outcome": {
                    "agent": self.agent_name,
                    "result": result,
                    "status": "success",
                },
                "llm_calls": get_llm_calls(self.state),
                "messages": messages_out,
                "next_step_suggestion": None,
            }
            return dct
        except Exception as exc:
            if isinstance(exc, LLMCallLimitExceeded):
                raise
            console.print(f"[red]✗ Error executing {self.agent_name}: {str(exc)}[/red]")
            if is_verbose(self.state):
                console.print(f"[red]{traceback.format_exc()}[/red]")
            return {
                "agent_outcome": {
                    "agent": self.agent_name,
                    "result": str(exc),
                    "status": "error",
                },
                "llm_calls": get_llm_calls(self.state),
                "next_step_suggestion": None,
            }

    def _execute_llm_profile(self, task: str) -> tuple[list[BaseMessage], str]:
        profile_name = self.agent_name.split(":", 1)[1]
        if is_verbose(self.state):
            console.print(f"[dim]Using LLM profile: {profile_name}[/dim]")
        response = create_and_invoke(
            self.config,
            [HumanMessage(content=task)],
            profile_name=profile_name,
            temperature=self.config.llm.generation_temperature,
            invoked_by="execution.llm_profile",
            state=self.state,
        )
        result = response.content
        if is_verbose(self.state):
            console.print(f"[dim]LLM response length: {len(result)} chars[/dim]")
            console.print(f"[dim]LLM response preview: {result[:300]}...[/dim]")
        return [AIMessage(content=result)], result,

    def _execute_agent(self, task: str) -> tuple[list[BaseMessage], str]:
        agent_config = self.config.agents.configs.get(self.resolved_agent)
        if agent_config and getattr(agent_config, "type", None) == "llm":
            # TODO: extract messages_out from LLM agent execution
            return [], self._execute_llm_agent(agent_config, task)
        return self._execute_registered_agent(task)

    def _execute_llm_agent(self, agent_config, task: str) -> str:
        model = agent_config.config.get("model")
        matching_profile = None
        for profile_name, profile in self.config.llm.profiles.items():
            if profile.model == model:
                matching_profile = profile_name
                break

        response = create_and_invoke(
            self.config,
            [HumanMessage(content=task)],
            profile_name=matching_profile,
            temperature=self.config.llm.generation_temperature,
            invoked_by="execution.llm_agent",
            state=self.state,
        )
        result = response.content
        if is_verbose(self.state):
            console.print(f"[dim]LLM response length: {len(result)} chars[/dim]")
            console.print(f"[dim]LLM response preview: {result[:300]}...[/dim]")
        return result

    def _execute_registered_agent(self, task: str) -> tuple[list[BaseMessage], str]:
        agent_class = AgentRegistry.get(self.resolved_agent)
        if is_verbose(self.state):
            console.print(f"[dim]Using agent class: {agent_class.__name__ if agent_class else 'None'}[/dim]")
        if not agent_class:
            available = ", ".join(AgentRegistry.list_agents())
            raise ValueError(f"Unknown agent: {self.resolved_agent}. Available agents: {available}")

        agent = agent_class(self.config, override_config=self.profile_override)

        # Wrap agent in tool-using loop for iterative execution
        return self._execute_agent_with_tools(agent, task)

    # TODO: make use of create_and_invoke_with_tool_support
    def _execute_agent_with_tools(self, agent, task: str, max_tool_iterations: int = 5) -> tuple[list[BaseMessage], str]:
        """Execute agent with automatic tool call detection and execution.

        This implements a tool-using loop where:
        1. Agent generates response
        2. Response is checked for tool calls
        3. Tools are executed and results fed back
        4. Loop continues until no more tool calls or max iterations
        """
        # Lazy import to break circular dependency at module load time
        from codur.graph.tool_detection import create_default_tool_detector
        tool_detector = create_default_tool_detector()

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

            if is_verbose(self.state):
                console.print(f"[dim]Agent iteration {tool_iteration}: result length {len(result)} chars[/dim]")

            # Check for tool calls in the response
            tool_calls = tool_detector.detect(result)

            if not tool_calls:
                # No tool calls detected, return final result
                if is_verbose(self.state):
                    console.print(f"[dim]Agent finished after {tool_iteration} iterations[/dim]")
                return result

            # Execute tools and collect results
            if is_verbose(self.state):
                console.print(f"[yellow]Detected {len(tool_calls)} tool call(s), executing...[/yellow]")

            execution = execute_tool_calls(tool_calls, self.state, self.config, augment=False, summary_mode="full")
            messages.extend(execution.messages)

            tool_iteration += 1

        if is_verbose(self.state):
            console.print(f"[yellow]Max tool iterations ({max_tool_iterations}) reached[/yellow]")
        return messages, result

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

