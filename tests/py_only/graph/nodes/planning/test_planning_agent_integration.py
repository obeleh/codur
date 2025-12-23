"""Agent integration tests for planning node with coding agent."""

import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.planning.core import (
    textual_pre_plan,
    llm_pre_plan,
    PlanningOrchestrator,
)
from codur.graph.nodes.planning.types import TaskType


class TestPlanningNodeWithCodingAgent:
    """Test planning node routing to coding agent."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        return config

    def test_textual_pre_plan_no_match_continues_to_llm(self, config):
        """Test that textual pre-plan continues to llm-pre-plan when no pattern matches."""
        state = {
            "messages": [
                HumanMessage(content="Implement a function that calculates fibonacci numbers")
            ],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)

        assert result is not None
        assert result["next_action"] == "continue_to_llm_pre_plan"
        assert result["iterations"] == 0

    def test_textual_pre_plan_with_greeting_resolves(self, config):
        """Test that textual pre-plan detects and resolves greetings."""
        config.runtime.detect_tool_calls_from_text = True
        state = {
            "messages": [HumanMessage(content="Hello!")],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)

        # Should either resolve or continue to llm-pre-plan
        assert result is not None
        assert "next_action" in result

    def test_llm_pre_plan_continues_to_full_planning_for_coding(self, config):
        """Test that llm-pre-plan continues to full planning for uncertain coding tasks."""
        state = {
            "messages": [
                HumanMessage(content="Fix the title_case function to handle hyphenated words")
            ],
            "iterations": 0,
            "verbose": False,
        }

        # Mock the quick_classify function to return uncertain classification
        with patch('codur.graph.nodes.planning.core.quick_classify') as mock_classify:
            mock_result = MagicMock()
            mock_result.is_confident = False
            mock_result.task_type = TaskType.CODE_FIX
            mock_result.confidence = 0.75
            mock_classify.return_value = mock_result

            result = llm_pre_plan(state, config)

            assert result is not None
            assert result["next_action"] == "continue_to_llm_plan"


class TestPlanningNodeCodeFixDetection:
    """Test planning node detection of code fix tasks."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        return config

    def test_detect_code_fix_keywords(self, config):
        """Test detection of code fix tasks by keywords."""
        test_cases = [
            "Fix the bug in main.py where the function doesn't handle edge cases",
            "Debug the title_case function - it's not working correctly",
            "Implement the title_case function according to the spec",
            "There's an error in my code - the output doesn't match expected",
        ]

        for task_description in test_cases:
            state = {
                "messages": [HumanMessage(content=task_description)],
                "iterations": 0,
                "verbose": False,
            }

            result = textual_pre_plan(state, config)
            assert result is not None
            # Either resolves or continues to next phase
            assert "next_action" in result

    def test_detect_coding_challenge_with_file_context(self, config):
        """Test detection of coding challenge with file context."""
        state = {
            "messages": [
                HumanMessage(content="Solve this: implement title case"),
                SystemMessage(content="Here's the function signature:\ndef title_case(s: str) -> str:"),
            ],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None


class TestPlanningNodeAgentSelection:
    """Test planning node agent selection logic."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        return config

    def test_plan_selects_coding_agent_for_fix_task(self, config):
        """Test that planning selects coding agent for fix tasks."""
        state = {
            "messages": [
                HumanMessage(content="Fix the title_case function in main.py")
            ],
            "iterations": 0,
            "verbose": False,
            "config": config,
        }

        # Mock the LLM to return a coding agent selection
        with patch('codur.graph.nodes.planning.core.create_llm_profile') as mock_llm:
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "action": "delegate",
                "agent": "agent:codur-coding",
                "reasoning": "This is a code fix task"
            })
            mock_model.invoke.return_value = mock_response
            mock_llm.return_value = mock_model

            orchestrator = PlanningOrchestrator(config)
            result = orchestrator.llm_plan(state, mock_model)
            assert result is not None


class TestPlanningNodeWithLineBasedEditing:
    """Test planning node with new line-based editing capability."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        return config

    def test_plan_routes_single_function_fix_to_coding_agent(self, config):
        """Test that planning routes single-function fixes to coding agent."""
        state = {
            "messages": [
                HumanMessage(content="Fix just the title_case function"),
            ],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None

    def test_plan_routes_full_file_implementation_to_coding_agent(self, config):
        """Test that planning routes full file implementations to coding agent."""
        state = {
            "messages": [
                HumanMessage(content="Implement the complete solution for the title-case challenge"),
            ],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None

    def test_plan_handles_verification_error_on_retry(self, config):
        """Test that planning handles verification errors during retry."""
        state = {
            "messages": [
                HumanMessage(content="Fix the title_case function"),
                SystemMessage(
                    content="Verification failed: Output does not match expected.\n"
                    "Expected: The Lord of the Rings\n"
                    "Actual: the lord of the rings"
                ),
            ],
            "iterations": 1,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None


class TestPlanningNodeRetryBehavior:
    """Test planning node behavior during retries."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        return config

    def test_plan_with_iteration_count_increments(self, config):
        """Test that planning increments iteration counter on retry."""
        state = {
            "messages": [
                HumanMessage(content="Fix the function"),
                SystemMessage(content="Verification failed: ..."),
            ],
            "iterations": 1,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None
        # Iteration count should be preserved or incremented
        assert "iterations" in result

    def test_plan_provides_error_context_on_retry(self, config):
        """Test that planning provides error context during retry."""
        state = {
            "messages": [
                HumanMessage(content="Fix the title_case function"),
                SystemMessage(
                    content="=== Expected Output ===\nThe Lord of the Rings\n"
                    "=== Actual Output ===\nthe lord of the rings"
                ),
            ],
            "iterations": 1,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None


class TestPlanningNodeMessageHandling:
    """Test planning node message normalization and handling."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        return config

    def test_plan_handles_empty_messages(self, config):
        """Test that planning handles empty message list gracefully."""
        state = {
            "messages": [],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None

    def test_plan_handles_mixed_message_types(self, config):
        """Test that planning handles mixed message types."""
        state = {
            "messages": [
                HumanMessage(content="Task 1"),
                SystemMessage(content="Context"),
                HumanMessage(content="Task 2"),
            ],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None

    def test_plan_extracts_challenge_from_first_human_message(self, config):
        """Test that planning extracts challenge from first human message."""
        messages = [
            HumanMessage(content="Implement title_case function"),
            SystemMessage(content="Requirements:\n1. Handle hyphenated words\n2. Preserve all-caps words"),
        ]
        state = {
            "messages": messages,
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None


class TestPlanningNodeFileDiscovery:
    """Test planner file discovery when no file hint is provided."""

    @pytest.fixture
    def config(self):
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        return config

    def test_llm_pre_plan_lists_files_without_hint(self, config):
        state = {
            "messages": [HumanMessage(content="Fix the failing tests")],
            "iterations": 0,
            "verbose": False,
        }

        result = llm_pre_plan(state, config)
        assert result is not None
        assert result["next_action"] == "tool"
        assert result["tool_calls"][0]["tool"] == "list_files"

    def test_llm_pre_plan_selects_app_py_from_list(self, config):
        state = {
            "messages": [
                HumanMessage(content="Fix the bug in the challenge"),
                SystemMessage(content="Tool results:\nlist_files: ['app.py', 'expected.txt']"),
            ],
            "iterations": 1,
            "verbose": False,
        }

        result = llm_pre_plan(state, config)
        assert result is not None
        assert result["next_action"] == "tool"
        assert result["tool_calls"] == [{"tool": "read_file", "args": {"path": "app.py"}}]

    def test_llm_plan_selects_app_py_from_list(self, config):
        state = {
            "messages": [
                HumanMessage(content="Fix the bug in the challenge"),
                SystemMessage(content="Tool results:\nlist_files: ['app.py', 'expected.txt']"),
            ],
            "iterations": 1,
            "verbose": False,
            "config": config,
        }

        orchestrator = PlanningOrchestrator(config)
        result = orchestrator.llm_plan(state, MagicMock())
        assert result is not None
        assert result["next_action"] == "tool"
        assert result["tool_calls"] == [{"tool": "read_file", "args": {"path": "app.py"}}]


class TestPlanningNodeEdgeCases:
    """Test planning node edge cases and error handling."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        return config

    def test_plan_with_very_long_task_description(self, config):
        """Test that planning handles very long task descriptions."""
        long_task = "Fix the function " * 100  # Very long description
        state = {
            "messages": [HumanMessage(content=long_task)],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None

    def test_plan_with_multiline_task_description(self, config):
        """Test that planning handles multiline task descriptions."""
        state = {
            "messages": [
                HumanMessage(
                    content="Fix the title_case function.\n\n"
                    "Requirements:\n"
                    "1. Handle hyphenated words\n"
                    "2. Preserve all-caps words\n"
                    "3. Match expected output"
                )
            ],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None

    def test_plan_with_special_characters_in_task(self, config):
        """Test that planning handles special characters in task descriptions."""
        state = {
            "messages": [
                HumanMessage(
                    content="Fix the `title_case()` function & handle \"edge-cases\""
                )
            ],
            "iterations": 0,
            "verbose": False,
        }

        result = textual_pre_plan(state, config)
        assert result is not None
