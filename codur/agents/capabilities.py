"""Agent capability registry and task matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from codur.config import CodurConfig


@dataclass(frozen=True)
class AgentCapabilities:
    name: str
    multi_file_support: bool
    file_system_access: bool
    supported_languages: List[str]
    max_context_size: int
    streaming_support: bool
    tool_access: bool
    estimated_cost: str
    offline_capable: bool


@dataclass(frozen=True)
class TaskRequirements:
    requires_multi_file: bool
    requires_file_system: bool
    language: Optional[str]
    requires_streaming: bool
    cost_sensitive: bool
    requires_offline: bool


AGENT_CAPABILITIES: Dict[str, AgentCapabilities] = {
    "claude_code": AgentCapabilities(
        name="Claude Code",
        multi_file_support=True,
        file_system_access=True,
        supported_languages=["python", "javascript", "go", "rust", "java"],
        max_context_size=200_000,
        streaming_support=False,
        tool_access=True,
        estimated_cost="expensive",
        offline_capable=False,
    ),
    "codex": AgentCapabilities(
        name="Codex",
        multi_file_support=False,
        file_system_access=True,
        supported_languages=["python", "javascript", "go", "rust", "java"],
        max_context_size=128_000,
        streaming_support=False,
        tool_access=True,
        estimated_cost="cheap",
        offline_capable=False,
    ),
    "ollama": AgentCapabilities(
        name="Ollama",
        multi_file_support=False,
        file_system_access=True,
        supported_languages=["python", "javascript", "go", "rust", "java"],
        max_context_size=8_000,
        streaming_support=True,
        tool_access=True,
        estimated_cost="free",
        offline_capable=True,
    ),
}


def extract_task_requirements(task: str) -> TaskRequirements:
    lowered = task.lower()
    return TaskRequirements(
        requires_multi_file=_has_keyword(lowered, ["refactor", "multi-file", "module", "system", "codebase"]),
        requires_file_system=_has_keyword(lowered, ["create", "delete", "modify", "write", "update", "edit"]),
        language=_detect_language(lowered),
        requires_streaming=_has_keyword(lowered, ["long task", "stream", "streaming", "progress"]),
        cost_sensitive=_has_keyword(lowered, ["free", "offline", "budget", "cheap"]),
        requires_offline=_has_keyword(lowered, ["offline", "no api", "local"]),
    )


def score_agent_match(task_req: TaskRequirements, capabilities: AgentCapabilities) -> float:
    score = 100.0

    if task_req.requires_multi_file and not capabilities.multi_file_support:
        return 0.0
    if task_req.requires_offline and not capabilities.offline_capable:
        return 0.0

    if task_req.cost_sensitive and capabilities.estimated_cost == "expensive":
        score -= 20.0
    if task_req.requires_streaming and not capabilities.streaming_support:
        score -= 15.0
    if task_req.requires_file_system and not capabilities.file_system_access:
        score -= 25.0
    if task_req.language and task_req.language not in capabilities.supported_languages:
        score -= 10.0

    return max(score, 0.0)


def select_best_agent(task: str, config: CodurConfig) -> tuple[str, float, Dict[str, float]]:
    task_req = extract_task_requirements(task)
    scores: Dict[str, float] = {}

    for agent_name, agent_config in config.agents.configs.items():
        if not agent_config.enabled or agent_config.type == "mcp":
            continue
        capabilities = AGENT_CAPABILITIES.get(agent_name)
        if not capabilities:
            continue
        scores[agent_name] = score_agent_match(task_req, capabilities)

    if not scores:
        default_agent = config.agents.preferences.default_agent
        return default_agent, 0.0, {}

    best_agent = max(scores.items(), key=lambda item: item[1])
    return best_agent[0], best_agent[1], scores


def _has_keyword(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _detect_language(text: str) -> Optional[str]:
    for language in ["python", "javascript", "typescript", "go", "rust", "java"]:
        if language in text:
            return language
    return None
