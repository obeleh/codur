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

from codur.graph.nodes.planning.types import (
    TaskType,
    ClassificationResult,
    ClassificationCandidate,
)


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
    "repair", "fail", "fails", "failed", "failing", "failure"
}

EXPLAIN_KEYWORDS = {
    "what does", "explain", "describe", "how does", "tell me about",
    "what is", "summarize", "summary", "how to use",
}

COMPLEX_KEYWORDS = {
    "refactor", "redesign", "migrate", "restructure", "rewrite",
    "multiple files", "entire", "all files", "codebase"
}

GENERATION_KEYWORDS = {
    "write", "create", "add", "generate", "make", "build", "new",
    "implement", "complete", "finish", "solve"
}

WEB_SEARCH_KEYWORDS = {
    "weather", "search", "latest", "news", "today", "current", "how is", "what are",
    "who is", "when did", "stock", "price", "market", "google", "find"
}

BROAD_QUESTION_KEYWORDS = {
    "who", "when", "where", "why", "what"
}

CODE_CONTEXT_KEYWORDS = {
    "traceback", "stack", "exception", "test", "tests", "unit test",
    "log", "logging", "debug", "print", "printf", "function", "method", "class",
    "module", "import", "lint", "format", "typing", "type", "edge case",
    "script", "cli", "tool", "automation", "dashboard", "ui", "interface",
    "api", "service", "app", "workflow", "flow", "pipeline", "scheduler",
    "cron", "report", "reports", "frontend", "backend", "code",
}

WEB_STRONG_KEYWORDS = {
    "weather", "news", "price", "stock", "market", "bitcoin", "forecast",
}

LOOKUP_KEYWORDS = {
    "find", "search", "locate", "usage", "used", "reference", "references",
    "where", "who", "when",
}

CODE_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php",
    ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".kt",
}


def extract_file_paths(text: str) -> list[str]:
    """Extract file paths from text."""
    paths = []

    # @file.py syntax
    at_matches = re.findall(r"@([^\s,]+)", text)
    paths.extend(at_matches)

    # Explicit file extensions
    ext_matches = re.findall(r"([^\s,\"']+\.(?:json|yaml|html|css|yml|txt|py|js|ts|md))", text)
    paths.extend(ext_matches)

    # Quoted paths
    quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
    for q in quoted:
        if "/" in q or "." in q:
            paths.append(q)

    # Clean paths: strip leading '@' and duplicates
    cleaned_paths = {p.lstrip('@') for p in paths}

    return list(cleaned_paths)


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _build_candidates(
    scores: dict[TaskType, float],
    reasons: dict[TaskType, list[str]],
) -> list[ClassificationCandidate]:
    candidates: list[ClassificationCandidate] = []
    for task_type in TaskType:
        score = scores.get(task_type, 0.0)
        confidence = min(0.49, 0.4 + (score * 0.05))
        reason_list = reasons.get(task_type, [])
        reasoning = "; ".join(reason_list) if reason_list else "baseline"
        candidates.append(
            ClassificationCandidate(
                task_type=task_type,
                confidence=confidence,
                reasoning=reasoning,
            )
        )
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates


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
        candidates = _build_candidates({}, {task: [] for task in TaskType})
        return ClassificationResult(
            task_type=TaskType.UNKNOWN,
            confidence=0.4,
            detected_files=[],
            detected_action=None,
            reasoning="No user message found",
            candidates=candidates,
        )

    text = user_message.strip()
    text_lower = text.lower()
    words = set(text_lower.split())

    # Extract file paths
    detected_files = extract_file_paths(text)
    has_code_file = any(Path(path).suffix.lower() in CODE_FILE_EXTENSIONS for path in detected_files)

    has_greeting = text_lower in GREETING_PATTERNS or (len(words) <= 3 and words & GREETING_PATTERNS)
    has_fix = any(kw in text_lower for kw in FIX_KEYWORDS)
    has_generation = any(kw in text_lower for kw in GENERATION_KEYWORDS)
    has_explain = any(kw in text_lower for kw in EXPLAIN_KEYWORDS)
    has_complex = any(kw in text_lower for kw in COMPLEX_KEYWORDS)
    has_web = any(kw in text_lower for kw in WEB_SEARCH_KEYWORDS)
    has_broad_question = any(kw in words for kw in BROAD_QUESTION_KEYWORDS)
    has_code_context = has_code_file or any(kw in text_lower for kw in CODE_CONTEXT_KEYWORDS)
    lookup_intent = any(kw in text_lower for kw in LOOKUP_KEYWORDS)
    project_hint = any(kw in text_lower for kw in ("project", "codebase", "repo", "repository"))
    explicit_code_request = any(
        kw in text_lower
        for kw in ("code", "script", "function", "class", "module", "api", "cli", "library", "program")
    )
    if any(kw in text_lower for kw in ("dashboard", "ui", "interface", "frontend", "backend")):
        explicit_code_request = True
    strong_code_context = has_code_file or explicit_code_request
    has_other_intent = has_fix or has_generation or has_explain or has_complex or has_web or bool(detected_files)

    scores = {
        TaskType.GREETING: 0.0,
        TaskType.FILE_OPERATION: 0.0,
        TaskType.CODE_FIX: 0.0,
        TaskType.CODE_GENERATION: 0.0,
        TaskType.EXPLANATION: 0.0,
        TaskType.COMPLEX_REFACTOR: 0.0,
        TaskType.WEB_SEARCH: 0.0,
        TaskType.UNKNOWN: 0.0,
    }
    reasons: dict[TaskType, list[str]] = {task: [] for task in scores}

    def bump(task: TaskType, amount: float, reason: str) -> None:
        if amount <= 0:
            return
        scores[task] += amount
        reasons[task].append(reason)

    if has_greeting:
        bump(TaskType.GREETING, 0.4, "greeting keyword")
        if not has_other_intent:
            bump(TaskType.GREETING, 0.3, "no other intent signals")

    file_op_action: str | None = None
    file_op_action_score = 0.0
    file_listing_intent = any(
        phrase in text_lower
        for phrase in ("list files", "list all files", "show files", "show all files", "list directory")
    )
    if file_listing_intent:
        bump(TaskType.FILE_OPERATION, 0.85, "file listing intent")
        file_op_action = "list_files"
        file_op_action_score = 0.85
    refactor_language = any(
        kw in text_lower
        for kw in (
            "behavior", "logic", "rule", "rules", "function", "method", "class", "module",
            "code", "implementation", "cache", "caching", "log", "logs", "logging",
            "validation", "parsing", "processing", "handler", "workflow", "flow", "pipeline",
        )
    )
    file_op_code_intent = False
    multi_file_refactor = len(detected_files) > 1 and refactor_language
    for keyword, action in FILE_OP_KEYWORDS.items():
        if not detected_files or not _contains_word(text_lower, keyword):
            continue
        penalty = 0.0
        if has_code_context:
            penalty += 0.4
        if has_fix or has_generation or has_explain:
            penalty += 0.2
        if keyword in ("rename", "remove") and has_code_context:
            penalty += 0.2
        if refactor_language:
            penalty += 0.35
        if detected_files and not has_code_file:
            penalty = max(0.0, penalty - 0.4)
        score = max(0.0, 0.9 - penalty)
        bump(TaskType.FILE_OPERATION, score, f"file operation keyword: {keyword}")
        if score > file_op_action_score:
            file_op_action = action
            file_op_action_score = score
        if refactor_language:
            file_op_code_intent = True

    if has_complex:
        base_complex = 0.7
        if has_explain or lookup_intent:
            base_complex = 0.35
        bump(TaskType.COMPLEX_REFACTOR, base_complex, "complex refactor keyword")
        if len(detected_files) > 1:
            bump(TaskType.COMPLEX_REFACTOR, 0.1, "multiple files hinted")
    if multi_file_refactor:
        bump(TaskType.COMPLEX_REFACTOR, 0.75, "multi-file refactor cues")

    if has_fix:
        bump(TaskType.CODE_FIX, 0.65, "fix/debug keyword")
        if detected_files:
            bump(TaskType.CODE_FIX, 0.2, "file hint present")
        if has_code_context:
            bump(TaskType.CODE_FIX, 0.1, "code context cues")
        if has_web:
            bump(TaskType.CODE_FIX, 0.2, "fix request with web terms")
    if file_op_code_intent:
        bump(TaskType.CODE_FIX, 0.2 if multi_file_refactor else 0.4, "file-op verb with code intent")

    if has_generation:
        bump(TaskType.CODE_GENERATION, 0.6, "generation keyword")
        if not has_fix:
            bump(TaskType.CODE_GENERATION, 0.05, "no fix keywords")
        if not detected_files:
            bump(TaskType.CODE_GENERATION, 0.05, "no file hint")
        if strong_code_context:
            bump(TaskType.CODE_GENERATION, 0.15, "code artifact context")

    if has_explain:
        bump(TaskType.EXPLANATION, 0.55, "explanation keyword")
        if detected_files:
            bump(TaskType.EXPLANATION, 0.2, "file hint present")
        if has_broad_question and has_code_context:
            bump(TaskType.EXPLANATION, 0.1, "question about code context")
    if lookup_intent and (project_hint or has_code_context or detected_files):
        bump(TaskType.EXPLANATION, 0.45, "code lookup request")
    if lookup_intent and not (project_hint or has_code_context or detected_files) and not has_web:
        bump(TaskType.EXPLANATION, 0.3, "lookup question without web intent")
    if has_broad_question and has_code_context and not has_explain:
        bump(TaskType.EXPLANATION, 0.35, "broad question with code context")

    for keyword in WEB_SEARCH_KEYWORDS:
        if keyword in text_lower:
            bump(TaskType.WEB_SEARCH, 0.12, f"web keyword: {keyword}")
            if not detected_files and not explicit_code_request:
                bump(TaskType.WEB_SEARCH, 0.08, "no local code context")
    for keyword in WEB_STRONG_KEYWORDS:
        if keyword in text_lower:
            bump(TaskType.WEB_SEARCH, 0.18, f"strong web keyword: {keyword}")
    if has_web and not detected_files and not explicit_code_request:
        bump(TaskType.WEB_SEARCH, 0.25, "web intent without code context")
    if has_broad_question and not detected_files and not has_code_context:
        bump(TaskType.WEB_SEARCH, 0.2, "broad question without code context")

    if detected_files and has_code_context and not (has_fix or has_generation or has_explain or has_complex):
        bump(TaskType.CODE_FIX, 0.3, "file hint with code context")

    priority = {
        TaskType.CODE_FIX: 6,
        TaskType.CODE_GENERATION: 5,
        TaskType.COMPLEX_REFACTOR: 4,
        TaskType.EXPLANATION: 3,
        TaskType.FILE_OPERATION: 2,
        TaskType.WEB_SEARCH: 1,
        TaskType.GREETING: 0,
    }
    best_task = max(scores.items(), key=lambda item: (item[1], priority.get(item[0], 0)))[0]
    best_score = scores[best_task]

    candidates = _build_candidates(scores, reasons)
    if best_score <= 0.0:
        return ClassificationResult(
            task_type=TaskType.UNKNOWN,
            confidence=0.4,
            detected_files=detected_files,
            detected_action=None,
            reasoning="No clear pattern matched",
            candidates=candidates,
        )

    detected_action = None
    if best_task == TaskType.GREETING:
        detected_action = "respond"
    elif best_task == TaskType.FILE_OPERATION:
        detected_action = file_op_action
    elif best_task == TaskType.EXPLANATION:
        detected_action = "read_file" if detected_files else None
    elif best_task == TaskType.WEB_SEARCH:
        detected_action = "duckduckgo_search"
    else:
        detected_action = "delegate"

    reasoning = "; ".join(reasons.get(best_task) or []) or "No clear pattern matched"
    best_confidence = next(
        (candidate.confidence for candidate in candidates if candidate.task_type == best_task),
        0.4,
    )

    return ClassificationResult(
        task_type=best_task,
        confidence=best_confidence,
        detected_files=detected_files,
        detected_action=detected_action,
        reasoning=reasoning,
        candidates=candidates,
    )
