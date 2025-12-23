"""Quick task classifier for two-phase planning.

Phase 1: Fast classification without LLM call
Phase 2: Full LLM planning only when needed, with context-aware prompt
"""

from __future__ import annotations

import re
from typing import Optional
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage

from codur.config import CodurConfig

from codur.graph.nodes.planning.types import TaskType, ClassificationResult


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
