# Task Strategies Package

This package contains task-specific strategies for the planning system. Each strategy owns the domain knowledge for one task type:

- **Discovery behavior** (Phase 0) - File listing, reading, etc.
- **Prompt building** (Phase 2) - Task-specific LLM planning prompts

## How to Create a New Strategy

To add a new strategy for a task type:

1.  **Define the TaskType**: If it's a new task type, add it to `TaskType` enum in `codur/graph/nodes/planning/types.py`.
2.  **Create a New File**: Create a new Python file in this directory (e.g., `my_new_task.py`).
3.  **Implement the Strategy**: Create a class that implements the `TaskStrategy` protocol defined in `base.py`.

```python
from codur.graph.nodes.planning.strategies.base import TaskStrategy
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
        # Return PlanNodeResult to resolve in Phase 0
        # Return None to pass to next phase
        return None

    def build_planning_prompt(
        self,
        classification: ClassificationResult,
        config: CodurConfig,
    ) -> str:
        # Return a context-aware planning prompt for this task type
        return "..."
```

4.  **Register the Strategy**: Import and add your strategy instance to the `_STRATEGIES` dictionary in `__init__.py`.

## Directory Structure

- `base.py`: Defines the `TaskStrategy` interface.
- `__init__.py`: Registry and `get_strategy_for_task()` dispatcher.
- `[task_type].py`: Task-specific implementations.
- `prompt_utils.py`: Shared prompt building utilities.

## Design Principles

- **Speed**: Strategies should be fast and avoid LLM calls if possible.
- **Discovery**: Use strategies to gather context (e.g., `list_files`) if information is missing.
- **Conservative**: Phase 0 does discovery only; Phase 2 makes routing decisions.
- **Domain Ownership**: Each strategy is the single source of truth for its task type.
