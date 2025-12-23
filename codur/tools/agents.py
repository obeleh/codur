"""Agent execution tools."""

from typing import Optional

from langchain_core.messages import HumanMessage

from codur.config import CodurConfig
from codur.llm import create_llm_profile
from codur.graph.state import AgentState
from codur.agents.ollama_agent import OllamaAgent
from codur.agents.codex_agent import CodexAgent
from codur.agents.claude_code_agent import ClaudeCodeAgent
from codur.utils.llm_calls import invoke_llm


def _resolve_agent_reference(raw_agent: str) -> str:
    if raw_agent.startswith("agent:"):
        return raw_agent.split(":", 1)[1]
    return raw_agent


def _resolve_agent_profile(config: CodurConfig, agent_name: str) -> tuple[str, Optional[dict]]:
    if agent_name.startswith("agent:"):
        profile_name = agent_name.split(":", 1)[1]
        profile = config.agents.profiles.get(profile_name)
        if profile and profile.name:
            return profile.name, profile.config
    return agent_name, None


def agent_call(
    agent: str,
    challenge: str,
    file_path: str = "",
    file_contents: str = "",
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> str:
    """Invoke an agent with a coding challenge and optional file context.

    Used by the planner to directly invoke agents with file contents after reading them.

    Args:
        agent: Agent name or reference (agent:<name> or llm:<profile>)
        challenge: The coding challenge/task description
        file_path: Path to the file (for context/reference)
        file_contents: The file contents to include as context
        config: Codur configuration for agent setup

    Returns:
        Agent response string
    """
    if not agent:
        raise ValueError("agent_call requires an 'agent' argument")
    if not challenge:
        raise ValueError("agent_call requires a 'challenge' argument")

    if config is None:
        raise ValueError("Config not available in tool state")

    # Build the task with file context if available
    task = challenge
    if file_contents:
        task = f"{challenge}\n\n=== File: {file_path} ===\n{file_contents}"

    agent_name, profile_override = _resolve_agent_profile(config, agent)
    resolved_agent = _resolve_agent_reference(agent_name)

    if agent_name.startswith("llm:"):
        profile_name = agent_name.split(":", 1)[1]
        llm = create_llm_profile(config, profile_name, temperature=config.llm.generation_temperature)
        response = invoke_llm(
            llm,
            [HumanMessage(content=task)],
            invoked_by="tools.agent_call",
            state=state,
            config=config,
        )
        return response.content
    if resolved_agent == "ollama":
        agent_instance = OllamaAgent(config, override_config=profile_override)
        return agent_instance.execute(task)
    if resolved_agent == "codex":
        agent_instance = CodexAgent(config, override_config=profile_override)
        return agent_instance.execute(task)
    if resolved_agent == "claude_code":
        agent_instance = ClaudeCodeAgent(config, override_config=profile_override)
        return agent_instance.execute(task)
    if resolved_agent == "codur-coding":
        # Special handling for codur-coding agent - invoke the actual coding node
        from codur.graph.nodes.coding import coding_node  # type: ignore
        from pathlib import Path

        tool_calls = [{"tool": "read_file", "args": {"path": file_path}}] if file_path else []
        if file_path.endswith(".py"):
            tool_calls.append({"tool": "python_ast_dependencies", "args": {"path": file_path}})

        # Build minimal state for coding node
        coding_state = {
            "messages": [HumanMessage(content=task)],
            "iterations": 0,
            "verbose": state.get("verbose", False) if state else False,
            "config": config,
            "llm_calls": int(state.get("llm_calls", 0)) if state else 0,
            "max_llm_calls": state.get("max_llm_calls") if state else config.runtime.max_llm_calls,
            "tool_calls": tool_calls
        }

        # Invoke coding node
        result_dict = coding_node(coding_state, config)  # type: ignore
        if state is not None:
            state["llm_calls"] = coding_state.get("llm_calls", state.get("llm_calls", 0))
        result = result_dict["agent_outcome"]["result"]

        return result

    raise ValueError(f"Unknown agent: {agent_name}")


def retry_in_agent(
    agent: str,
    task: str,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> str:
    """Retry a task using a specific agent name.

    Args:
        agent: Agent name or reference (agent:<name> or llm:<profile>)
        task: Task content to run
        config: Codur configuration for agent setup

    Returns:
        Agent response string
    """
    if not agent:
        raise ValueError("retry_in_agent requires an 'agent' argument")
    if not task:
        raise ValueError("retry_in_agent requires a 'task' argument")

    if config is None:
        if state is not None and hasattr(state, "get_config"):
            config = state.get_config()
        if config is None:
            raise ValueError("Config not available in tool state")

    agent_name, profile_override = _resolve_agent_profile(config, agent)
    resolved_agent = _resolve_agent_reference(agent_name)

    if agent_name.startswith("llm:"):
        profile_name = agent_name.split(":", 1)[1]
        llm = create_llm_profile(config, profile_name, temperature=config.llm.generation_temperature)
        response = invoke_llm(
            llm,
            [HumanMessage(content=task)],
            invoked_by="tools.retry_in_agent",
            state=state,
            config=config,
        )
        return response.content
    if resolved_agent == "ollama":
        agent_instance = OllamaAgent(config, override_config=profile_override)
        return agent_instance.execute(task)
    if resolved_agent == "codex":
        agent_instance = CodexAgent(config, override_config=profile_override)
        return agent_instance.execute(task)
    if resolved_agent == "claude_code":
        agent_instance = ClaudeCodeAgent(config, override_config=profile_override)
        return agent_instance.execute(task)

    raise ValueError(f"Unknown agent: {agent_name}")
