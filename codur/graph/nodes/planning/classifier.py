"""Quick task classifier for two-phase planning.

Phase 1: Fast classification without LLM call
Phase 2: Full LLM planning only when needed, with context-aware prompt

This classifier delegates scoring to individual TaskStrategy implementations,
keeping domain knowledge where it belongs.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage

from codur.config import CodurConfig

from codur.graph.nodes.planning.types import (
    TaskType,
    ClassificationResult,
    ClassificationCandidate,
)
from codur.graph.nodes.planning.strategies import (
    GreetingStrategy,
    FileOperationStrategy,
    CodeFixStrategy,
    CodeGenerationStrategy,
    ExplanationStrategy,
    ComplexRefactorStrategy,
    WebSearchStrategy,
    UnknownStrategy,
)

# Code file extensions for detection
CODE_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php",
    ".cs", ".cpp", ".c", ".h", ".hpp", ".swift", ".kt",
}

# Strategy instances for each task type
_STRATEGIES = {
    TaskType.GREETING: GreetingStrategy(),
    TaskType.FILE_OPERATION: FileOperationStrategy(),
    TaskType.CODE_FIX: CodeFixStrategy(),
    TaskType.CODE_GENERATION: CodeGenerationStrategy(),
    TaskType.EXPLANATION: ExplanationStrategy(),
    TaskType.COMPLEX_REFACTOR: ComplexRefactorStrategy(),
    TaskType.WEB_SEARCH: WebSearchStrategy(),
    TaskType.UNKNOWN: UnknownStrategy(),
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


def _build_candidates(
    scores: dict[TaskType, float],
    reasons: dict[TaskType, list[str]],
) -> list[ClassificationCandidate]:
    candidates: list[ClassificationCandidate] = []
    for task_type in TaskType:
        score = scores.get(task_type, 0.0)
        confidence = min(0.95, 0.4 + (score * 0.5))
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


def _determine_detected_action(
    task_type: TaskType,
    text_lower: str,
    detected_files: list[str],
) -> str | None:
    """Determine the detected action based on task type and context."""
    if task_type == TaskType.GREETING:
        return "respond"
    elif task_type == TaskType.FILE_OPERATION:
        # Check for file listing intent
        file_listing_phrases = {"list files", "list all files", "show files", "show all files", "list directory"}
        if any(phrase in text_lower for phrase in file_listing_phrases):
            return "list_files"
        # Check for file operation keywords
        file_op_strategy = _STRATEGIES[TaskType.FILE_OPERATION]
        patterns = file_op_strategy.get_patterns()
        for keyword, action in patterns.action_keywords.items():
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                return action
        return None
    elif task_type == TaskType.EXPLANATION:
        return "read_file" if detected_files else None
    elif task_type == TaskType.WEB_SEARCH:
        return "duckduckgo_search"
    else:
        return "delegate"


def quick_classify(messages: list[BaseMessage], config: CodurConfig) -> ClassificationResult:
    """Quickly classify task without LLM call.

    This is Phase 1 of two-phase planning. It uses pattern matching to
    classify obvious cases with high confidence, allowing us to skip
    the expensive LLM call for simple tasks.

    Scoring is delegated to individual TaskStrategy implementations.
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

    # Delegate scoring to strategies
    scores: dict[TaskType, float] = {}
    reasons: dict[TaskType, list[str]] = {}

    for task_type, strategy in _STRATEGIES.items():
        contribution = strategy.compute_score(text_lower, words, detected_files, has_code_file)
        scores[task_type] = contribution.score
        reasons[task_type] = contribution.reasoning

    # Priority for tie-breaking
    priority = {
        TaskType.CODE_FIX: 6,
        TaskType.CODE_GENERATION: 5,
        TaskType.COMPLEX_REFACTOR: 4,
        TaskType.EXPLANATION: 3,
        TaskType.FILE_OPERATION: 2,
        TaskType.WEB_SEARCH: 1,
        TaskType.GREETING: 0,
    }
    if config.verbose:
        print("Quick Classifier Scores and Reasons:")
        for task, score in scores.items():
            reason_text = "; ".join(reasons.get(task) or []) or "No reasons"
            print(f"  {task.value}: {score:.2f} ({reason_text})")
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

    detected_action = _determine_detected_action(best_task, text_lower, detected_files)
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
