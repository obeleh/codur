"""Agent selection helpers for planning hints."""

from __future__ import annotations

from codur.config import CodurConfig
from codur.agents.capabilities import select_best_agent


_CHALLENGE_KEYWORDS = (
    "coding challenge",
    "challenge",
    "prompt.txt",
    "expected.txt",
    "docstring",
    "requirements",
)

_MULTIFILE_KEYWORDS = (
    "multi-file",
    "multiple files",
    "codebase",
    "project-wide",
    "entire project",
    "entire codebase",
)


def select_agent_for_task(
    config: CodurConfig,
    user_message: str,
    detected_files: list[str],
    routing_key: str | None = None,
    prefer_multifile: bool = False,
    allow_coding_agent: bool = False,
) -> str:
    """Select an agent based on routing preferences and task signals."""
    user_message = user_message or ""
    if allow_coding_agent and _looks_like_coding_challenge(user_message, detected_files):
        return "agent:codur-coding"

    routing = config.agents.preferences.routing or {}
    if prefer_multifile or _looks_like_multifile(user_message, detected_files):
        multifile_agent = routing.get("multifile")
        if multifile_agent:
            return multifile_agent

    if routing_key:
        routed_agent = routing.get(routing_key)
        if routed_agent:
            return routed_agent

    default_agent = config.agents.preferences.default_agent
    if default_agent:
        return default_agent

    best_agent, _, _ = select_best_agent(user_message, config)
    if best_agent:
        return f"agent:{best_agent}"

    return "agent:codur-coding"


def _looks_like_coding_challenge(user_message: str, detected_files: list[str]) -> bool:
    lowered = user_message.lower()
    if any(keyword in lowered for keyword in _CHALLENGE_KEYWORDS):
        return True
    for path in detected_files:
        lower_path = path.lower()
        if "challenges/" in lower_path or lower_path.endswith(("prompt.txt", "expected.txt")):
            return True
    return False


def _looks_like_multifile(user_message: str, detected_files: list[str]) -> bool:
    lowered = user_message.lower()
    if any(keyword in lowered for keyword in _MULTIFILE_KEYWORDS):
        return True
    return len(detected_files) > 1
