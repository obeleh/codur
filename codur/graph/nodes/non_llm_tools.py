"""Non-LLM planning helpers that can directly produce actions."""

from __future__ import annotations

from typing import Optional
import os
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from rich.console import Console

from codur.graph.state import AgentState
from codur.graph.nodes.types import PlanNodeResult
from codur.tools.filesystem import EXCLUDE_DIRS
from codur.constants import GREETING_MAX_WORDS
from codur.graph.nodes.tool_detection import create_default_tool_detector


console = Console()


def _trivial_response(text: str) -> Optional[str]:
    stripped = text.strip().lower()
    if not stripped:
        return "Hi! How can I help you with your coding tasks today?"
    if "thank" in stripped or "thanks" in stripped:
        return "You're welcome! Anything else you want to tackle?"
    greetings = {
        "hi", "hello", "hey", "yo", "sup",
        "good morning", "good afternoon", "good evening",
    }
    if stripped in greetings:
        return "Hi! How can I help you with your coding tasks today?"
    if len(stripped.split()) <= GREETING_MAX_WORDS and any(word in greetings for word in stripped.split()):
        return "Hi! How can I help you with your coding tasks today?"
    return None


def _looks_like_explain_request(text: str) -> bool:
    lowered = text.lower()
    triggers = ("what does", "explain", "describe", "summarize", "summary of")
    return any(trigger in lowered for trigger in triggers)


def _find_workspace_match(raw_text: str) -> Optional[str]:
    text = raw_text.strip()
    if not text:
        return None
    candidates = []
    for token in text.replace('"', " ").replace("'", " ").split():
        if token.startswith("@") or ".py" in token or "/" in token or "\\" in token:
            cleaned = token.strip(".,:;()[]{}")
            if cleaned.startswith("@"):
                cleaned = cleaned[1:]
            if cleaned:
                candidates.append(cleaned)
    cwd = Path.cwd()
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = cwd / path

        try:
            if path.is_absolute() and not path.resolve().is_relative_to(cwd.resolve()):
                continue
        except (ValueError, OSError):
            continue

        if path.exists():
            try:
                return str(path.relative_to(cwd) if path.is_relative_to(cwd) else path)
            except ValueError:
                continue

        if path.name == candidate and not ("/" in candidate or "\\" in candidate):
            matches = []
            for root, dirnames, filenames in os.walk(cwd):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
                if candidate in filenames:
                    matches.append(Path(root) / candidate)
                if len(matches) > 1:
                    break
            if len(matches) == 1:
                match = matches[0]
                try:
                    return str(match.relative_to(cwd))
                except ValueError:
                    continue
    return None


_TOOL_DETECTOR = create_default_tool_detector()


def run_non_llm_tools(messages: list[BaseMessage], state: AgentState) -> Optional[PlanNodeResult]:
    tool_results_present = any(
        isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
        for msg in messages
    )

    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    verbose = state.get("verbose", False)

    if last_human_msg:
        trivial_response = _trivial_response(last_human_msg)
        if trivial_response:
            if verbose:
                console.log("[dim]Non-LLM tool node: Detected trivial response.[/dim]")
            return {
                "next_action": "end",
                "final_response": trivial_response,
                "iterations": state.get("iterations", 0) + 1,
            }

    if last_human_msg and not tool_results_present:
        if _looks_like_explain_request(last_human_msg):
            matched_path = _find_workspace_match(last_human_msg)
            if matched_path:
                if verbose:
                    console.log(f"[dim]Non-LLM tool node: Detected explain request for file {matched_path}.[/dim]")
                return {
                    "next_action": "tool",
                    "tool_calls": [
                        {"tool": "read_file", "args": {"path": matched_path}},
                        {"tool": "agent_call", "args": {"challenge": f"Please explain the contents of the file {matched_path}.", "agent": "agent:codur-explaining"}},
                    ],
                    "iterations": state.get("iterations", 0) + 1,
                }

        file_op_result = _TOOL_DETECTOR.detect(last_human_msg)
        if file_op_result:
            if verbose:
                console.log(f"[dim]Non-LLM tool node: Detected file operation tools: {file_op_result}.[/dim]")
            return {
                "next_action": "tool",
                "tool_calls": file_op_result,
                "iterations": state.get("iterations", 0) + 1,
            }

    return None
