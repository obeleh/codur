"""Simple execution metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json


@dataclass
class ExecutionMetrics:
    task_id: str
    timestamp: str
    agent_selected: str
    iterations: int
    status: str
    duration_seconds: Optional[float]
    verification_passed: Optional[bool]


class MetricsCollector:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metrics_path = root / ".codur" / "metrics.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def record_execution(self, metrics: ExecutionMetrics) -> None:
        payload = asdict(metrics)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def build_metrics(
    *,
    task_id: str,
    agent_selected: str,
    iterations: int,
    status: str,
    duration_seconds: Optional[float],
    verification_passed: Optional[bool],
) -> ExecutionMetrics:
    return ExecutionMetrics(
        task_id=task_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_selected=agent_selected,
        iterations=iterations,
        status=status,
        duration_seconds=duration_seconds,
        verification_passed=verification_passed,
    )
