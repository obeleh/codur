"""Tests for LLM classification phase."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from codur.graph.planning.phases.llm_classification_phase import llm_classification


@pytest.fixture
def config():
    config = MagicMock()
    config.planning = MagicMock()
    config.planning.use_llm_pre_plan = True
    config.llm = MagicMock()
    config.llm.default_profile = "test-profile"
    config.verbose = False
    return config


def test_llm_classification_skips_when_disabled(config):
    config.planning.use_llm_pre_plan = False
    classification = MagicMock()
    state = {
        "messages": [HumanMessage(content="Hello")],
        "iterations": 0,
        "verbose": False,
        "classification": classification,
    }

    result = llm_classification(state, config)

    assert result["next_action"] == "continue_to_llm_plan"
    assert result["classification"] is classification


def test_llm_classification_responds_on_high_confidence(config):
    response = MagicMock()
    response.content = json.dumps({
        "task_type": "greeting",
        "confidence": 0.9,
        "detected_files": [],
        "suggested_action": "respond",
        "reasoning": "Simple greeting",
    })
    state = {
        "messages": [HumanMessage(content="Hello")],
        "iterations": 0,
        "verbose": False,
    }

    with patch(
        "codur.graph.planning.phases.llm_classification_phase.create_and_invoke",
        return_value=response,
    ):
        result = llm_classification(state, config)

    assert result["next_action"] == "end"
    assert result["final_response"] == "Simple greeting"
    assert result["llm_debug"]["phase1_llm_resolved"] is True


def test_llm_classification_continues_on_error(config):
    classification = MagicMock()
    state = {
        "messages": [HumanMessage(content="Fix the bug")],
        "iterations": 0,
        "verbose": False,
        "classification": classification,
    }

    with patch(
        "codur.graph.planning.phases.llm_classification_phase.create_and_invoke",
        side_effect=RuntimeError("boom"),
    ):
        result = llm_classification(state, config)

    assert result["next_action"] == "continue_to_llm_plan"
    assert result["classification"] is classification
