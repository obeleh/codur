"""Shared planning types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, FrozenSet


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


@dataclass(frozen=True)
class ClassificationCandidate:
    """Candidate classification with confidence and reasoning."""

    task_type: TaskType
    confidence: float
    reasoning: str


@dataclass
class ClassificationResult:
    """Result from quick classification."""

    task_type: TaskType
    confidence: float  # 0.0 to 1.0
    detected_files: list[str]
    detected_action: Optional[str]
    reasoning: str
    candidates: list[ClassificationCandidate] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        """Return True if classification is confident enough to skip full planning."""
        return self.confidence >= 0.8


@dataclass(frozen=True)
class PatternConfig:
    """Pattern configuration for a task strategy.

    Each strategy defines its own patterns for classification.
    The classifier uses these to compute scores.
    """

    # Primary keywords that strongly indicate this task type
    primary_keywords: FrozenSet[str] = field(default_factory=frozenset)

    # Secondary keywords that boost confidence when combined with primary
    boosting_keywords: FrozenSet[str] = field(default_factory=frozenset)

    # Keywords that reduce confidence (indicate different task type)
    negative_keywords: FrozenSet[str] = field(default_factory=frozenset)

    # File extensions this task type typically works with
    file_extensions: FrozenSet[str] = field(default_factory=frozenset)

    # Action keywords that map to specific tool actions (e.g., {"move": "move_file"})
    action_keywords: dict[str, str] = field(default_factory=dict)

    # Phrases that indicate this task type (for multi-word matching)
    phrases: FrozenSet[str] = field(default_factory=frozenset)


@dataclass
class ScoreContribution:
    """Score contribution from a strategy's pattern matching."""

    score: float
    reasoning: list[str] = field(default_factory=list)

    def add(self, amount: float, reason: str) -> None:
        """Add to the score with a reason."""
        if amount > 0:
            self.score += amount
            self.reasoning.append(reason)
