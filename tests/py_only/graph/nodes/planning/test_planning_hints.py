"""Tests for planning hints strategies."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import SystemMessage, HumanMessage

from codur.graph.nodes.planning.hints.web_search import WebSearchStrategy
from codur.graph.nodes.planning.hints.greeting import GreetingStrategy
from codur.graph.nodes.planning.hints.file_operation import FileOperationStrategy
from codur.graph.nodes.planning.hints.code_fix import CodeFixStrategy
from codur.graph.nodes.planning.types import ClassificationResult, TaskType

class TestPlanningHints:
    def test_web_search_strategy(self):
        """Test web search strategy."""
        strategy = WebSearchStrategy()
        classification = MagicMock(spec=ClassificationResult)
        classification.task_type = TaskType.WEB_SEARCH
        
        messages = [HumanMessage(content="Find weather in SF")]
        
        # Case 1: Web search task, no tool results -> trigger search
        result = strategy.execute(
            classification=classification,
            tool_results_present=False,
            messages=messages,
            iterations=0,
            config=MagicMock(),
            verbose=False
        )
        assert result is not None
        assert result["next_action"] == "tool"
        assert result["tool_calls"][0]["tool"] == "duckduckgo_search"
        assert result["tool_calls"][0]["args"]["query"] == "Find weather in SF"

    def test_greeting_strategy(self):
        """Test greeting strategy."""
        strategy = GreetingStrategy()
        classification = MagicMock(spec=ClassificationResult)
        classification.task_type = TaskType.GREETING
        
        result = strategy.execute(
            classification=classification,
            tool_results_present=False,
            messages=[],
            iterations=0,
            config=MagicMock(),
            verbose=False
        )
        assert result is not None
        assert result["next_action"] == "end"
        assert "Hello" in result["final_response"]

    def test_file_operation_strategy(self):
        """Test file operation strategy."""
        strategy = FileOperationStrategy()
        classification = MagicMock(spec=ClassificationResult)
        classification.task_type = TaskType.FILE_OPERATION
        classification.detected_action = "delete_file"
        classification.detected_files = ["test.py"]
        
        result = strategy.execute(
            classification=classification,
            tool_results_present=False,
            messages=[],
            iterations=0,
            config=MagicMock(),
            verbose=False
        )
        assert result is not None
        assert result["next_action"] == "tool"
        assert result["tool_calls"][0]["tool"] == "delete_file"

    def test_code_fix_no_file_hint(self):
        """Test code fix strategy listing files when no hint is present."""
        strategy = CodeFixStrategy()
        classification = MagicMock(spec=ClassificationResult)
        classification.task_type = TaskType.CODE_FIX
        classification.detected_files = []
        
        result = strategy.execute(
            classification=classification,
            tool_results_present=False,
            messages=[],
            iterations=0,
            config=MagicMock(),
            verbose=False
        )
        assert result is not None
        assert result["next_action"] == "tool"
        assert result["tool_calls"][0]["tool"] == "list_files"

    @patch("codur.graph.nodes.planning.hints.code_fix.tool_results_include_read_file")
    @patch("codur.graph.nodes.planning.hints.code_fix.select_file_from_tool_results")
    def test_code_fix_file_selection(self, mock_select, mock_check_read):
        """Test code fix strategy selecting file from tool results."""
        strategy = CodeFixStrategy()
        mock_check_read.return_value = False
        mock_select.return_value = "main.py"
        
        classification = MagicMock(spec=ClassificationResult)
        classification.task_type = TaskType.CODE_FIX
        classification.detected_files = []

        result = strategy.execute(
            classification=classification,
            tool_results_present=True,
            messages=[],
            iterations=0,
            config=MagicMock(),
            verbose=False
        )
        assert result is not None
        assert result["next_action"] == "tool"
        assert result["tool_calls"][0]["tool"] == "read_file"
        assert result["tool_calls"][0]["args"]["path"] == "main.py"
