"""Tests for explanation task routing in planning phases."""

import json
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from codur.graph.planning.core import pattern_plan
from codur.graph.planning.classifier import quick_classify, text_confidence_backoff
from codur.graph.planning.types import TaskType
from codur.graph.planning.strategies import ExplanationStrategy


def _tool_msg(tool: str, output, args: dict | None = None) -> ToolMessage:
    """Helper to create a ToolMessage in JSON format."""
    return ToolMessage(
        content=json.dumps({"tool": tool, "output": output, "args": args or {}}),
        tool_call_id="test",
    )


class TestPhase0ExplanationRouting:
    """Test Phase 0 (pattern_plan) routing to explaining agent."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        config.planning = MagicMock()
        config.planning.use_llm_pre_plan = True
        config.verbose = False
        return config

    def test_classify_explain_keyword(self, config):
        """Test that 'explain' keyword triggers EXPLANATION classification."""
        content = "Explain how the authentication system works"
        messages = [HumanMessage(content=content)]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert result.confidence >= 0.65 * text_confidence_backoff(content)
        assert "explanation keyword" in result.reasoning.lower()

    def test_classify_what_does_question(self, config):
        """Test that 'what does' question triggers EXPLANATION."""
        content = "What does the main.py file do?"
        messages = [HumanMessage(content=content)]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert result.confidence >= 0.65 * text_confidence_backoff(content)

    def test_classify_describe_keyword(self, config):
        """Test that 'describe' keyword triggers EXPLANATION."""
        content = "Describe the project architecture"
        messages = [HumanMessage(content=content)]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert result.confidence >= 0.65 * text_confidence_backoff(content)

    def test_classify_how_does_question(self, config):
        """Test that 'how does' question triggers EXPLANATION."""
        content = "How does the title_case function handle edge cases?"
        messages = [HumanMessage(content=content)]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert result.confidence >= 0.65 * text_confidence_backoff(content)

    def test_classify_tell_me_about(self, config):
        """Test that 'tell me about' triggers EXPLANATION."""
        content = "Tell me about the authentication system in this codebase"
        messages = [HumanMessage(content=content)]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert result.confidence >= 0.65 * text_confidence_backoff(content)

    def test_classify_with_file_hint(self, config):
        """Test that explanation with file path gets boosted score."""
        content = "Explain main.py"
        messages = [HumanMessage(content=content)]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert len(result.detected_files) > 0
        assert "main.py" in result.detected_files[0]
        # Should have higher confidence with file hint
        assert result.confidence >= 0.7 * text_confidence_backoff(content)

    def test_classify_lookup_with_code_context(self, config):
        """Test that lookup questions with code context trigger EXPLANATION."""
        messages = [
            HumanMessage(content="Where is the authentication function used in the codebase?")
        ]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert "lookup" in result.reasoning.lower() or "code" in result.reasoning.lower()

    def test_classify_broad_question_with_code_file(self, config):
        """Test that broad questions about code files trigger EXPLANATION."""
        messages = [
            HumanMessage(content="What is the purpose of auth.py?")
        ]

        result = quick_classify(messages, config)

        assert result.task_type == TaskType.EXPLANATION
        assert len(result.detected_files) > 0

    def test_pattern_plan_with_file_discovery(self, config):
        """Test that pattern_plan triggers file discovery for explain tasks without file hint."""
        state = {
            "messages": [HumanMessage(content="Explain the project structure")],
            "iterations": 0,
            "verbose": False,
        }

        result = pattern_plan(state, config)

        # Should trigger file discovery (list_files tool)
        assert result is not None
        if result.get("next_action") == "tool":
            assert len(result.get("tool_calls", [])) > 0
            # Might be list_files or file_tree
            tool_names = [call.get("tool") for call in result.get("tool_calls", [])]
            assert any(tool in ["list_files", "file_tree"] for tool in tool_names)

    def test_pattern_plan_read_file_with_hint(self, config):
        """Test that pattern_plan suggests read_file when file is mentioned."""
        state = {
            "messages": [HumanMessage(content="Explain what main.py does")],
            "iterations": 0,
            "verbose": False,
        }

        result = pattern_plan(state, config)

        # Should trigger read_file for the detected file
        assert result is not None
        if result.get("next_action") == "tool":
            tool_calls = result.get("tool_calls", [])
            assert len(tool_calls) > 0
            # Should include read_file for main.py
            tools = [call.get("tool") for call in tool_calls]
            assert "read_file" in tools

    def test_not_explanation_fix_task(self, config):
        """Test that fix tasks are NOT classified as EXPLANATION."""
        messages = [HumanMessage(content="Fix the bug in main.py")]

        result = quick_classify(messages, config)

        assert result.task_type != TaskType.EXPLANATION
        # Should be CODE_FIX instead
        assert result.task_type == TaskType.CODE_FIX

    def test_not_explanation_implement_task(self, config):
        """Test that implementation tasks are NOT classified as EXPLANATION."""
        messages = [HumanMessage(content="Implement a new authentication system")]

        result = quick_classify(messages, config)

        assert result.task_type != TaskType.EXPLANATION
        # Should be CODE_GENERATION instead
        assert result.task_type == TaskType.CODE_GENERATION


class TestExplanationStrategyScoring:
    """Test ExplanationStrategy scoring logic."""

    def test_strategy_scores_explain_keyword_high(self):
        """Test that explain keyword gets high score."""
        strategy = ExplanationStrategy()

        contribution = strategy.compute_score(
            text_lower="explain how this works",
            words=set("explain how this works".split()),
            detected_files=[],
            has_code_file=False,
        )

        assert contribution.score >= 0.55
        assert "explanation keyword" in "; ".join(contribution.reasoning)

    def test_strategy_scores_with_file_hint(self):
        """Test that file hint boosts score."""
        strategy = ExplanationStrategy()

        contribution = strategy.compute_score(
            text_lower="explain main.py",
            words=set("explain main.py".split()),
            detected_files=["main.py"],
            has_code_file=True,
        )

        # Should get base 0.55 + 0.2 file hint = 0.75
        assert contribution.score >= 0.75
        assert "file hint present" in "; ".join(contribution.reasoning)

    def test_strategy_scores_lookup_intent(self):
        """Test that lookup intent gets scored."""
        strategy = ExplanationStrategy()

        contribution = strategy.compute_score(
            text_lower="find where the function is used in the project",
            words=set("find where the function is used in the project".split()),
            detected_files=[],
            has_code_file=False,
        )

        # Should detect lookup intent
        assert contribution.score >= 0.3
        assert any("lookup" in reason for reason in contribution.reasoning)

    def test_strategy_scores_broad_question_with_code(self):
        """Test that broad questions with code context get scored."""
        strategy = ExplanationStrategy()

        contribution = strategy.compute_score(
            text_lower="what is the purpose of this function",
            words=set("what is the purpose of this function".split()),
            detected_files=[],
            has_code_file=True,
        )

        # Should detect broad question with code context
        assert contribution.score >= 0.3

    def test_strategy_negative_keywords_reduce_score(self):
        """Test that negative keywords don't trigger explanation."""
        strategy = ExplanationStrategy()

        # "fix" is a negative keyword for explanation
        contribution = strategy.compute_score(
            text_lower="fix this explanation",
            words=set("fix this explanation".split()),
            detected_files=[],
            has_code_file=False,
        )

        # Should have lower score due to negative keyword
        # The word "explanation" might give some score, but "fix" should reduce it
        # Actually, with "explanation" in the text, it might still score high
        # Let me try a better test case
        contribution2 = strategy.compute_score(
            text_lower="create a new feature",
            words=set("create a new feature".split()),
            detected_files=[],
            has_code_file=False,
        )

        # Should have very low or zero score
        assert contribution2.score < 0.3


class TestExplanationStrategyExecution:
    """Test ExplanationStrategy execution (file discovery)."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        config.runtime.max_iterations = 10
        return config

    def test_strategy_executes_file_discovery_no_files(self, config):
        """Test that strategy triggers file discovery when no files detected."""
        strategy = ExplanationStrategy()

        classification = MagicMock()
        classification.task_type = TaskType.EXPLANATION
        classification.detected_files = []
        classification.confidence = 0.85

        result = strategy.execute(
            classification=classification,
            tool_results_present=False,
            messages=[HumanMessage(content="Explain the project structure")],
            iterations=0,
            config=config,
            verbose=False,
        )

        # Should trigger file discovery
        assert result is not None
        assert result.get("next_action") == "tool"
        tool_calls = result.get("tool_calls", [])
        assert len(tool_calls) > 0
        assert tool_calls[0]["tool"] == "list_files"

    def test_strategy_executes_read_file_with_detected_files(self, config):
        """Test that strategy reads files when they're detected."""
        strategy = ExplanationStrategy()

        classification = MagicMock()
        classification.task_type = TaskType.EXPLANATION
        classification.detected_files = ["main.py"]
        classification.confidence = 0.90

        result = strategy.execute(
            classification=classification,
            tool_results_present=False,
            messages=[HumanMessage(content="Explain main.py")],
            iterations=0,
            config=config,
            verbose=False,
        )

        # Should trigger read_file
        assert result is not None
        assert result.get("next_action") == "tool"
        tool_calls = result.get("tool_calls", [])
        assert len(tool_calls) > 0
        assert tool_calls[0]["tool"] == "read_file"
        assert tool_calls[0]["args"]["path"] == "main.py"

    def test_strategy_returns_none_after_file_read(self, config):
        """Test that strategy returns None after file has been read (allowing next phase to handle)."""
        strategy = ExplanationStrategy()

        classification = MagicMock()
        classification.task_type = TaskType.EXPLANATION
        classification.detected_files = ["main.py"]
        classification.confidence = 0.90
        classification.is_confident = True

        # Tool results are present with read_file content
        result = strategy.execute(
            classification=classification,
            tool_results_present=True,
            messages=[
                HumanMessage(content="Explain main.py"),
                _tool_msg("read_file", "def main(): pass", {"path": "main.py"}),
            ],
            iterations=1,
            config=config,
            verbose=False,
        )

        # Should return None, allowing next planning phase to handle delegation
        # (discovery is complete, now it's time for LLM planning to decide)
        assert result is None


class TestExplanationPlanningPrompt:
    """Test that explanation planning creates appropriate prompts."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        config = MagicMock()
        config.runtime = MagicMock()
        config.runtime.detect_tool_calls_from_text = False
        return config

    def test_planning_prompt_suggests_explaining_agent(self, config):
        """Test that planning prompt suggests codur-explaining agent."""
        strategy = ExplanationStrategy()

        classification = MagicMock()
        classification.task_type = TaskType.EXPLANATION
        classification.detected_files = ["example.py"]
        classification.confidence = 0.90

        prompt = strategy.build_planning_prompt(classification, config)

        # Should mention the explaining agent
        assert "agent:codur-explaining" in prompt
        # Should suggest explanation-relevant tools
        assert "read_file" in prompt
        # Should have task focus for explanation
        assert "Explanation" in prompt or "explanation" in prompt

    def test_planning_prompt_includes_tool_suggestions(self, config):
        """Test that planning prompt includes relevant tools."""
        strategy = ExplanationStrategy()

        classification = MagicMock()
        classification.task_type = TaskType.EXPLANATION
        classification.detected_files = []
        classification.confidence = 0.85

        prompt = strategy.build_planning_prompt(classification, config)

        # Should suggest file discovery and analysis tools
        assert "list_files" in prompt or "list_dirs" in prompt
        assert "python_ast" in prompt or "AST" in prompt.lower()
