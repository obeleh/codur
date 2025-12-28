"""Task strategies package for planning.

Each strategy owns the domain knowledge for one TaskType:
- Discovery behavior (Phase 0)
- Prompt building for LLM planning (Phase 2)
"""

from .base import TaskStrategy
from .greeting import GreetingStrategy
from .file_operation import FileOperationStrategy
from .code_fix import CodeFixStrategy
from .code_generation import CodeGenerationStrategy
from .explanation import ExplanationStrategy
from .complex_refactor import ComplexRefactorStrategy
from .web_search import WebSearchStrategy
from .unknown import UnknownStrategy

from codur.constants import TaskType

_STRATEGIES: dict[TaskType, TaskStrategy] = {
    TaskType.GREETING: GreetingStrategy(),
    TaskType.FILE_OPERATION: FileOperationStrategy(),
    TaskType.CODE_FIX: CodeFixStrategy(),
    TaskType.CODE_GENERATION: CodeGenerationStrategy(),
    TaskType.EXPLANATION: ExplanationStrategy(),
    TaskType.COMPLEX_REFACTOR: ComplexRefactorStrategy(),
    TaskType.WEB_SEARCH: WebSearchStrategy(),
    TaskType.UNKNOWN: UnknownStrategy(),
}


def get_strategy_for_task(task_type: TaskType) -> TaskStrategy:
    """Get the appropriate strategy for a given task type."""
    return _STRATEGIES.get(task_type, UnknownStrategy())
