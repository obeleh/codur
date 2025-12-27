# Planning strategies

This package contains task-specific strategies used by the planning system.

## Roles in planning

Strategies contribute in three places:

- Classification patterns and scoring for Phase 0 (quick classifier)
- Optional Phase 0 resolution via `execute` (fast discovery or short-circuit)
- Phase 2 prompt building via `build_planning_prompt`

Phase 1 LLM pre-plan is a separate classifier and does not use strategies.

## Creating a new strategy

1. Add a new `TaskType` in `codur/constants.py` if needed.
2. Create a file in this directory (for example, `my_task.py`).
3. Implement the `TaskStrategy` protocol from `base.py`:

```python
from codur.graph.planning.strategies.base import TaskStrategy
from codur.graph.node_types import PlanNodeResult
from codur.graph.planning.types import ClassificationResult
from codur.config import CodurConfig
from langchain_core.messages import BaseMessage


class MyTaskStrategy:
    def get_patterns(self):
        ...

    def compute_score(self, text_lower, words, detected_files, has_code_file):
        ...

    def execute(
            self,
            classification: ClassificationResult,
            tool_results_present: bool,
            messages: list[BaseMessage],
            iterations: int,
            config: CodurConfig,
            verbose: bool = False,
    ) -> PlanNodeResult | None:
        return None

    def build_planning_prompt(self, classification: ClassificationResult, config: CodurConfig) -> str:
        return "..."
```

4. Register the strategy in `codur/graph/nodes/planning/strategies/__init__.py` and in the classifier registry.

## Design principles

- Keep strategies deterministic and fast.
- Avoid LLM calls inside strategies.
- Use centralized utilities in `codur/graph/nodes/planning/strategies/prompt_utils.py` when possible.
- Phase 0 should only do lightweight discovery; defer complex reasoning to Phase 2.
