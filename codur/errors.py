"""Structured error classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(Enum):
    NETWORK = "network"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    RESOURCE_EXHAUSTED = "resource"
    EXECUTION_FAILURE = "execution"
    CONFIGURATION = "config"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CodurError:
    category: ErrorCategory
    message: str
    retry_eligible: bool
    backoff_multiplier: float = 2.0
    max_retries: int = 3
    agent_feedback: str = ""
    original_exception: Optional[Exception] = None

    def is_recoverable(self) -> bool:
        return self.retry_eligible

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "message": self.message,
            "retry_eligible": self.retry_eligible,
            "backoff_multiplier": self.backoff_multiplier,
            "max_retries": self.max_retries,
            "agent_feedback": self.agent_feedback,
        }


def classify_exception(exception: Exception) -> CodurError:
    message = str(exception)
    lowered = message.lower()

    if "timeout" in lowered:
        return CodurError(
            category=ErrorCategory.TIMEOUT,
            message=message,
            retry_eligible=True,
            backoff_multiplier=1.5,
            agent_feedback="Task took too long. Break it into smaller steps.",
            max_retries=2,
            original_exception=exception,
        )
    if "connection" in lowered or "max retries exceeded" in lowered:
        return CodurError(
            category=ErrorCategory.NETWORK,
            message=message,
            retry_eligible=True,
            backoff_multiplier=2.0,
            agent_feedback="Network error. Check connectivity and retry.",
            max_retries=3,
            original_exception=exception,
        )
    if "out of memory" in lowered or "memory" in lowered and "exceed" in lowered:
        return CodurError(
            category=ErrorCategory.RESOURCE_EXHAUSTED,
            message=message,
            retry_eligible=True,
            backoff_multiplier=3.0,
            agent_feedback="Resource exhausted. Reduce scope or use smaller input.",
            max_retries=1,
            original_exception=exception,
        )
    if "invalid" in lowered or "validation" in lowered:
        return CodurError(
            category=ErrorCategory.VALIDATION,
            message=message,
            retry_eligible=False,
            original_exception=exception,
        )

    return CodurError(
        category=ErrorCategory.UNKNOWN,
        message=message,
        retry_eligible=True,
        max_retries=1,
        original_exception=exception,
    )
