"""Tests for pattern-based planning phase."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from codur.constants import TaskType
from codur.graph.planning.phases.pattern_phase import pattern_plan
from codur.graph.planning.types import ClassificationResult


@pytest.fixture
def config():
    config = MagicMock()
    config.runtime = MagicMock()
    config.planning = MagicMock()
    config.runtime.detect_tool_calls_from_text = False
    config.planning.use_llm_pre_plan = True
    config.verbose = False
    return config


def test_pattern_phase_returns_non_llm_result(config):
    config.runtime.detect_tool_calls_from_text = True
    state = {
        "messages": [HumanMessage(content="hi")],
        "iterations": 0,
        "verbose": False,
    }
    expected = {"next_action": "end", "iterations": 1}

    with patch(
        "codur.graph.planning.phases.pattern_phase.run_non_llm_tools",
        return_value=expected,
    ) as mock_run:
        result = pattern_plan(state, config)

    assert result is expected
    mock_run.assert_called_once()


def test_pattern_phase_continues_when_strategy_returns_none(config):
    state = {
        "messages": [HumanMessage(content="Do something complex")],
        "iterations": 0,
        "verbose": False,
    }
    classification = ClassificationResult(
        task_type=TaskType.UNKNOWN,
        confidence=0.4,
        detected_files=[],
        detected_action=None,
        reasoning="test",
        candidates=[],
    )
    strategy = MagicMock()
    strategy.execute.return_value = None

    with patch(
        "codur.graph.planning.phases.pattern_phase.quick_classify",
        return_value=classification,
    ):
        with patch(
            "codur.graph.planning.phases.pattern_phase.get_strategy_for_task",
            return_value=strategy,
        ):
            result = pattern_plan(state, config)

    assert result["next_action"] == "continue_to_llm_classification"
    assert result["classification"] is classification
