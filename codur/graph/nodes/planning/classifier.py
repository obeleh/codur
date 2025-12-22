"""Quick task classifier for two-phase planning.

Phase 1: Fast classification without LLM call
Phase 2: Full LLM planning only when needed, with context-aware prompt
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage

from codur.config import CodurConfig
from codur.agents.capabilities import select_best_agent


class TaskType(Enum):
    """Task type classification for routing."""
    GREETING = "greeting"
    FILE_OPERATION = "file_operation"
    CODE_FIX = "code_fix"
    CODE_GENERATION = "code_generation"
    EXPLANATION = "explanation"
    COMPLEX_REFACTOR = "complex_refactor"
    WEB_SEARCH = "web_search"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result from quick classification."""
    task_type: TaskType
    confidence: float  # 0.0 to 1.0
    detected_files: list[str]
    detected_action: Optional[str]
    reasoning: str

    @property
    def is_confident(self) -> bool:
        """Return True if classification is confident enough to skip full planning."""
        return self.confidence >= 0.8


# Patterns for quick classification
GREETING_PATTERNS = {
    "hi", "hello", "hey", "yo", "sup", "thanks", "thank you",
    "good morning", "good afternoon", "good evening", "bye", "goodbye"
}

FILE_OP_KEYWORDS = {
    "move": "move_file",
    "copy": "copy_file",
    "delete": "delete_file",
    "remove": "delete_file",
    "rename": "move_file",
}

FIX_KEYWORDS = {
    "fix", "bug", "error", "debug", "issue", "broken", "incorrect", "wrong",
    "implement", "complete", "finish", "solve", "repair"
}

EXPLAIN_KEYWORDS = {
    "what does", "explain", "describe", "how does", "tell me about",
    "what is", "summarize", "summary"
}

COMPLEX_KEYWORDS = {
    "refactor", "redesign", "migrate", "restructure", "rewrite",
    "multiple files", "entire", "all files", "codebase"
}

GENERATION_KEYWORDS = {
    "write", "create", "add", "generate", "make", "build", "new"
}

WEB_SEARCH_KEYWORDS = {
    "weather", "search", "latest", "news", "today", "current", "how is", "what are",
    "who is", "when did", "stock", "price", "market", "google", "find"
}

BROAD_QUESTION_KEYWORDS = {
    "who", "when", "where", "why", "what"
}


def extract_file_paths(text: str) -> list[str]:
    """Extract file paths from text."""
    paths = []

    # @file.py syntax
    at_matches = re.findall(r"@([^\s,]+)", text)
    paths.extend(at_matches)

    # Explicit file extensions
    ext_matches = re.findall(r"([^\s,\"']+\.(?:py|js|ts|json|yaml|yml|md|txt|html|css))", text)
    paths.extend(ext_matches)

    # Quoted paths
    quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
    for q in quoted:
        if "/" in q or "." in q:
            paths.append(q)

    return list(set(paths))


def quick_classify(messages: list[BaseMessage], config: CodurConfig) -> ClassificationResult:
    """Quickly classify task without LLM call.

    This is Phase 1 of two-phase planning. It uses pattern matching to
    classify obvious cases with high confidence, allowing us to skip
    the expensive LLM call for simple tasks.
    """
    # Get last human message
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return ClassificationResult(
            task_type=TaskType.UNKNOWN,
            confidence=0.0,
            detected_files=[],
            detected_action=None,
            reasoning="No user message found"
        )

    text = user_message.strip()
    text_lower = text.lower()
    words = set(text_lower.split())

    # Extract file paths
    detected_files = extract_file_paths(text)

    # Check for greetings (highest confidence)
    if text_lower in GREETING_PATTERNS or (len(words) <= 3 and words & GREETING_PATTERNS):
        return ClassificationResult(
            task_type=TaskType.GREETING,
            confidence=0.95,
            detected_files=[],
            detected_action="respond",
            reasoning="Detected greeting pattern"
        )

    # Check for file operations
    for keyword, action in FILE_OP_KEYWORDS.items():
        if keyword in text_lower and detected_files:
            return ClassificationResult(
                task_type=TaskType.FILE_OPERATION,
                confidence=0.9,
                detected_files=detected_files,
                detected_action=action,
                reasoning=f"Detected file operation: {keyword}"
            )

    # Check for explanation requests
    if any(kw in text_lower for kw in EXPLAIN_KEYWORDS):
        if detected_files:
            return ClassificationResult(
                task_type=TaskType.EXPLANATION,
                confidence=0.85,
                detected_files=detected_files,
                detected_action="read_file",
                reasoning="Detected explanation request with file"
            )

    # Check for complex refactoring
    if any(kw in text_lower for kw in COMPLEX_KEYWORDS):
        return ClassificationResult(
            task_type=TaskType.COMPLEX_REFACTOR,
            confidence=0.8,
            detected_files=detected_files,
            detected_action="delegate",
            reasoning="Detected complex refactoring keywords"
        )

    # Check for fix/debug tasks
    if any(kw in text_lower for kw in FIX_KEYWORDS):
        confidence = 0.85 if detected_files else 0.7
        return ClassificationResult(
            task_type=TaskType.CODE_FIX,
            confidence=confidence,
            detected_files=detected_files,
            detected_action="delegate",
            reasoning="Detected fix/debug keywords"
        )

    # Check for code generation
    if any(kw in text_lower for kw in GENERATION_KEYWORDS):
        return ClassificationResult(
            task_type=TaskType.CODE_GENERATION,
            confidence=0.75,
            detected_files=detected_files,
            detected_action="delegate",
            reasoning="Detected code generation keywords"
        )

    # Check for web search (high confidence if specific keywords used and no files detected)
    if any(kw in text_lower for kw in WEB_SEARCH_KEYWORDS):
        if not detected_files:
            return ClassificationResult(
                task_type=TaskType.WEB_SEARCH,
                confidence=0.85,
                detected_files=[],
                detected_action="duckduckgo_search",
                reasoning="Detected web search/real-time keywords"
            )

    # Check for broad/ambiguous questions (low confidence web search)
    # This captures "Who won the game?" or "When is the deadline?"
    # Confidence is low (0.5) so it forces Phase 2 LLM planning, but provides the
    # WEB_SEARCH task type to prompt_builder for tool hints.
    if any(kw in words for kw in BROAD_QUESTION_KEYWORDS):
        if not detected_files:
            return ClassificationResult(
                task_type=TaskType.WEB_SEARCH,
                confidence=0.5,
                detected_files=[],
                detected_action=None,
                reasoning="Detected broad question pattern (possible web search)"
            )

    # Unknown - need full LLM planning
    return ClassificationResult(
        task_type=TaskType.UNKNOWN,
        confidence=0.3,
        detected_files=detected_files,
        detected_action=None,
        reasoning="No clear pattern matched"
    )


def get_agent_for_task_type(
    task_type: TaskType,
    config: CodurConfig,
    detected_files: list[str],
    user_message: str = ""
) -> str:
    """Get the appropriate agent for a task type using routing config and capability matching.

    First tries explicit routing config, then falls back to capability-based selection.
    """
    routing = config.agents.preferences.routing
    default_agent = config.agents.preferences.default_agent

    # Try explicit routing first
    if task_type == TaskType.COMPLEX_REFACTOR:
        # Use multifile agent for complex refactoring
        explicit_agent = routing.get("multifile", None)
        if explicit_agent:
            return explicit_agent

    if task_type == TaskType.CODE_FIX:
        # Use complex agent for debugging
        explicit_agent = routing.get("complex", None)
        if explicit_agent:
            return explicit_agent

    if task_type == TaskType.CODE_GENERATION:
        # Use simple agent for basic generation
        explicit_agent = routing.get("simple", None)
        if explicit_agent:
            return explicit_agent

    # Fallback to capability-based routing for smarter agent selection
    if user_message:
        best_agent, confidence, _ = select_best_agent(user_message, config)
        if confidence > 0:
            return best_agent

    # Final fallback to default agent
    return default_agent


def build_context_aware_prompt(
    classification: ClassificationResult,
    config: CodurConfig
) -> str:
    """Build a context-aware prompt for Phase 2 based on classification.

    This creates a focused prompt that's tailored to the detected task type,
    making the LLM's job easier and reducing token usage.
    """
    default_agent = config.agents.preferences.default_agent
    task_type = classification.task_type
    files_str = ", ".join(classification.detected_files) if classification.detected_files else "none detected"

    base_rules = f"""You are Codur. Respond with ONLY valid JSON.

Task Analysis:
- Detected type: {task_type.value}
- Confidence: {classification.confidence:.0%}
- Files mentioned: {files_str}
- Initial assessment: {classification.reasoning}

"""

    if task_type == TaskType.CODE_FIX:
        return base_rules + f"""This is a BUG FIX or IMPLEMENTATION task.

When deciding your response, consider:
1. Files mentioned: {files_str}
2. Is this fixing existing code or implementing from docstring?
3. Does the agent need file context to succeed?

ROUTING OPTIONS:
  Option A (Simple): Delegate directly to {default_agent}
    - Use when: No file context needed, straightforward fix
    - JSON: {{"action": "delegate", "agent": "{default_agent}", "reasoning": "...", "response": null}}

  Option B (With Context): Read file first, then use coding agent
    - Use when: Need to see file contents, implementation task, complex fix
    - JSON: {{"action": "tool", "agent": null, "reasoning": "...", "response": null, "tool_calls": [{{"tool": "read_file", "args": {{"path": "..."}}}}, {{"tool": "agent_call", "args": {{"agent": "agent:codur-coding", "challenge": "...", "file_path": "..."}}}}]}}

For file-based implementation tasks, prefer Option B.

Return ONLY valid JSON:"""

    if task_type == TaskType.CODE_GENERATION:
        return base_rules + f"""This is a CODE GENERATION task.

Consider: Does code generation need context from existing files?

ROUTING OPTIONS:
  Option A: Direct generation - {{"action": "delegate", "agent": "{default_agent}", "reasoning": "..."}}
  Option B: Read context first - {{"action": "tool", "tool_calls": [{{"tool": "read_file", ...}}, {{"tool": "agent_call", "args": {{"agent": "agent:codur-coding", ...}}}}]}}

Return ONLY valid JSON:"""

    if task_type == TaskType.COMPLEX_REFACTOR:
        multifile_agent = config.agents.preferences.routing.get("multifile", default_agent)
        return base_rules + f"""This is a COMPLEX REFACTORING task. Files: {files_str}

Delegate to multifile-capable agent or read files first for analysis.

{{"action": "delegate", "agent": "{multifile_agent}", "reasoning": "..."}} or
{{"action": "tool", "tool_calls": [{{"tool": "read_file", ...}}], ...}}

Return ONLY valid JSON:"""

    if task_type == TaskType.EXPLANATION:
        target_file = classification.detected_files[0] if classification.detected_files else 'unknown'
        return base_rules + f"""This is an EXPLANATION request. Target file: {target_file}

Must read file first to explain it.

{{"action": "tool", "agent": null, "reasoning": "read file for explanation", "tool_calls": [{{"tool": "read_file", "args": {{"path": "{target_file}"}}}}]}}

Return ONLY valid JSON:"""

    if task_type == TaskType.FILE_OPERATION:
        return base_rules + f"""This is a FILE OPERATION. Files: {files_str}

Tools: read_file, write_file, copy_file, move_file, delete_file, list_files

{{"action": "tool", "agent": null, "reasoning": "...", "tool_calls": [{{"tool": "...", "args": {{...}}}}]}}

Return ONLY valid JSON:"""

    if task_type == TaskType.WEB_SEARCH:
        return base_rules + f"""This task requires real-time information or general knowledge from the web.

Use duckduckgo_search to find information.

{{"action": "tool", "agent": null, "reasoning": "need real-time info", "tool_calls": [{{"tool": "duckduckgo_search", "args": {{"query": "..."}}}}]}}

Return ONLY valid JSON:"""

    # Unknown or needs full analysis
    return base_rules + f"""Task type uncertain. Analyze and choose appropriate action.

Options:
- "delegate" for code tasks: {{"action": "delegate", "agent": "{default_agent}", ...}}
- "tool" for file ops: {{"action": "tool", "tool_calls": [...], ...}}
- "tool" for web search (real-time info): {{"action": "tool", "tool_calls": [{{"tool": "duckduckgo_search", "args": {{"query": "..."}}}}]}}
- "respond" for greetings or direct answers: {{"action": "respond", "response": "...", ...}}

Return ONLY valid JSON:"""
