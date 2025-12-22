"""Task decomposition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Subtask:
    description: str


class TaskDecomposer:
    def decompose(self, task: str) -> List[Subtask]:
        if not self._is_complex(task):
            return [Subtask(task)]

        subtasks: List[Subtask] = []
        lowered = task.lower()
        if "refactor" in lowered:
            modules = self._extract_modules(task)
            if modules:
                for module in modules:
                    subtasks.append(Subtask(f"Refactor {module}"))
        if not subtasks:
            subtasks.append(Subtask(task))
        return subtasks

    def _is_complex(self, task: str) -> bool:
        word_count = len(task.split())
        keyword_count = sum(
            1 for keyword in ["multiple", "all", "entire", "system", "codebase", "multi-file"]
            if keyword in task.lower()
        )
        return word_count > 80 or keyword_count > 1

    def _extract_modules(self, task: str) -> List[str]:
        tokens = task.replace(",", " ").split()
        modules = [token for token in tokens if token.endswith((".py", "/"))]
        return modules
