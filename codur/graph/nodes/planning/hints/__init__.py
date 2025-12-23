"""Hints package for Phase 1 planning strategies."""

from .base import Phase1Strategy
from .greeting import GreetingStrategy
from .file_operation import FileOperationStrategy
from .code_fix import CodeFixStrategy
from .code_generation import CodeGenerationStrategy
from .explanation import ExplanationStrategy
from .complex_refactor import ComplexRefactorStrategy
from .web_search import WebSearchStrategy
from .unknown import UnknownStrategy

from codur.graph.nodes.planning.types import TaskType

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

def get_strategy_for_task(task_type: TaskType) -> Phase1Strategy:
    """Get the appropriate strategy for a given task type."""
    return _STRATEGIES.get(task_type, UnknownStrategy())
