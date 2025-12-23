"""Shared planning types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
