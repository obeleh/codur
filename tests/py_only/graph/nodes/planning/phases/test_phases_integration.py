"""Cross-phase planning tests."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from codur.constants import TaskType
from codur.graph.planning.phases.plan_phase import llm_plan
from codur.graph.planning.types import ClassificationResult


@pytest.fixture
def config():
    config = MagicMock()
    config.verbose = False
    return config


def test_phase2_uses_existing_classification(config):
    classification = ClassificationResult(
        task_type=TaskType.CODE_FIX,
        confidence=0.9,
        detected_files=[],
        detected_action=None,
        reasoning="test",
        candidates=[],
    )
    state = {
        "messages": [HumanMessage(content="Fix the bug")],
        "iterations": 0,
        "verbose": False,
        "config": config,
        "classification": classification,
    }

    result = llm_plan(config, MagicMock(), MagicMock(), MagicMock(), state, MagicMock())

    assert result["next_action"] == "tool"
    assert result["tool_calls"] == [{"tool": "list_files", "args": {}}]
