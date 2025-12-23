# Planning Hints Package

This package contains Phase 1 planning strategies (hints) for different task types. Phase 1 aims to resolve simple tasks or provide discovery steps (like listing files) before proceeding to full LLM planning in Phase 2.

## How to Create a New Hint/Strategy

To add a new strategy for a task type:

1.  **Define the TaskType**: If it's a new task type, add it to `TaskType` enum in `codur/graph/nodes/planning/types.py`.
2.  **Create a New File**: Create a new Python file in this directory (e.g., `my_new_task.py`).
3.  **Implement the Strategy**: Create a class that implements the `Phase1Strategy` protocol defined in `base.py`.

```python
from codur.graph.nodes.planning.hints.base import Phase1Strategy
from codur.graph.nodes.types import PlanNodeResult
# ... other imports

class MyNewTaskStrategy:
    def execute(
        self,
        classification: ClassificationResult,
        tool_results_present: bool,
        messages: list[BaseMessage],
        iterations: int,
        config: CodurConfig,
        verbose: bool = False
    ) -> PlanNodeResult | None:
        # Implementation logic here
        # Return PlanNodeResult to resolve Phase 1
        # Return None to pass to Phase 2
        return None

    def build_phase2_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        # Return a context-aware Phase 2 planning prompt for this task type
        return "..."
```

4.  **Register the Strategy**: Import and add your strategy instance to the `_STRATEGIES` dictionary in `__init__.py`.

## Directory Structure

- `base.py`: Defines the `Phase1Strategy` interface.
- `__init__.py`: Dispatcher logic to get the right strategy for a `TaskType`.
- `[task_type].py`: Task-specific implementations.

## Design Principles

- **Speed**: Strategies should be fast and avoid LLM calls if possible.
- **Discovery**: Use strategies to gather context (e.g., `list_files`) if information is missing.
- **Confidence**: Only resolve tasks in Phase 1 if the classification confidence is high (typically >= 0.8).
