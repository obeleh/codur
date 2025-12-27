"""Tests for complex refactor planning prompts."""

from types import SimpleNamespace

from codur.graph.nodes.planning.strategies.complex_refactor import ComplexRefactorStrategy
from codur.graph.nodes.planning.types import ClassificationResult, TaskType


def test_complex_refactor_prompt_suggests_rope_tools():
    strategy = ComplexRefactorStrategy()
    classification = ClassificationResult(
        task_type=TaskType.COMPLEX_REFACTOR,
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

    assert "rope_find_usages" in prompt
    assert "rope_find_definition" in prompt
    assert "rope_rename_symbol" in prompt
    assert "rope_move_module" in prompt
    assert "rope_extract_method" in prompt
