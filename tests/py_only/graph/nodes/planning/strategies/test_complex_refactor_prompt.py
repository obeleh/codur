"""Tests for complex refactor planning prompts."""

from types import SimpleNamespace

from codur.graph.planning.strategies.complex_refactor import ComplexRefactorStrategy
from codur.graph.planning.types import ClassificationResult, TaskType


def test_complex_refactor_prompt_suggests_investigation_tools():
    """Test that complex refactor strategy suggests investigation tools.

    After the tool-based refactor, the planner uses investigation tools
    and delegates to agents for mutations rather than using mutation tools directly.
    """
    strategy = ComplexRefactorStrategy()
    classification = ClassificationResult(
        task_type=TaskType.REFACTOR,
        confidence=0.9,
        detected_files=["app.py", "utils.py"],
        detected_action=None,
        reasoning="refactor request",
    )
    config = SimpleNamespace(
        agents=SimpleNamespace(
            preferences=SimpleNamespace(
                default_agent="agent:codur-coding",
                routing={},
            )
        )
    )

    prompt = strategy.build_planning_prompt(classification, config)

    # Planner should suggest investigation tools
    assert "list_files" in prompt
    assert "python_dependency_graph" in prompt
    assert "python_ast_dependencies_multifile" in prompt

    # Should guide towards delegation
    assert "delegate_task" in prompt
