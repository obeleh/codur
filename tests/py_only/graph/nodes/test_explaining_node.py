"""Tests for the explaining node."""

import json
from unittest.mock import MagicMock, patch

from codur.graph.explaining import explaining_node, _build_explaining_prompt
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


def _tool_msg(tool: str, output, args: dict | None = None) -> ToolMessage:
    """Helper to create a ToolMessage in JSON format."""
    return ToolMessage(
        content=json.dumps({"tool": tool, "output": output, "args": args or {}}),
        tool_call_id="test",
    )


class TestExplainingNode:
    """Test the explaining node."""

    def test_build_explaining_prompt_basic(self):
        """Test basic prompt construction."""
        messages = [
            HumanMessage(content="Explain main.py"),
            _tool_msg("read_file", "def main():\n    pass  # Content of main.py", {"path": "main.py"}),
            AIMessage(content="I see."),
        ]
        prompt = _build_explaining_prompt(messages)
        assert "EXPLANATION REQUEST:\nExplain main.py" in prompt
        assert "FILE CONTENTS:" in prompt
        assert "def main()" in prompt
        assert "Previous explanation: I see." in prompt

    def test_build_explaining_prompt_with_code(self):
        """Test prompt construction with code content."""
        messages = [
            HumanMessage(content="What does this function do?"),
            _tool_msg("read_file", "def title_case(s: str) -> str:\n    return s.title()", {"path": "utils.py"}),
        ]
        prompt = _build_explaining_prompt(messages)
        assert "EXPLANATION REQUEST:" in prompt
        assert "FILE CONTENTS:" in prompt
        assert "def title_case" in prompt

    def test_build_explaining_prompt_with_ast(self):
        """Test prompt construction with AST information."""
        messages = [
            HumanMessage(content="Explain the dependencies"),
            _tool_msg("python_ast_dependencies", "title_case -> str.title", {"path": "utils.py"}),
        ]
        prompt = _build_explaining_prompt(messages)
        assert "EXPLANATION REQUEST:" in prompt
        assert "CODE STRUCTURE (AST/Dependencies):" in prompt
        assert "title_case -> str.title" in prompt

    def test_build_explaining_prompt_with_tool_results(self):
        """Test prompt construction with tool results."""
        messages = [
            HumanMessage(content="Show me the project structure"),
            _tool_msg("list_files", ["main.py", "test.py", "README.md"]),
        ]
        prompt = _build_explaining_prompt(messages)
        assert "EXPLANATION REQUEST:" in prompt
        assert "TOOL RESULTS:" in prompt
        assert "list_files" in prompt

    def test_build_explaining_prompt_filters_verification_errors(self):
        """Test that verification errors are filtered out."""
        messages = [
            HumanMessage(content="Explain this"),
            SystemMessage(content="Verification failed: Output does not match"),
            SystemMessage(content="=== Expected Output ===\nSomething"),
            SystemMessage(content="Actual file content here"),
        ]
        prompt = _build_explaining_prompt(messages)
        assert "Verification failed" not in prompt
        assert "Expected Output" not in prompt
        assert "Actual file content here" in prompt

    @patch("codur.graph.explaining.create_and_invoke")
    def test_explaining_node_execution(self, mock_create_and_invoke):
        """Test that the node executes and returns a result."""
        # Setup mocks
        mock_response = MagicMock()
        mock_response.content = "This is the explanation."
        mock_create_and_invoke.return_value = mock_response

        # Setup state
        state = {
            "messages": [HumanMessage(content="Explain this.")],
            "iterations": 0,
            "verbose": False,
            "llm_calls": 0,
        }
        config = MagicMock()
        config.llm.profiles = {}
        config.llm.default_profile = "default"

        # Run node
        result = explaining_node(state, config)

        # Verify
        assert result["agent_outcomes"][0]["agent"] == "agent:codur-explaining"
        assert result["agent_outcomes"][0]["result"] == "This is the explanation."
        assert result["agent_outcomes"][0]["status"] == "success"

        # Verify LLM call with correct temperature
        mock_create_and_invoke.assert_called_once()
        call_kwargs = mock_create_and_invoke.call_args[1]
        assert call_kwargs["temperature"] == 0.5, "Should use temperature 0.5 for precision"

        # Verify system prompt
        args, kwargs = mock_create_and_invoke.call_args
        messages = args[1]
        assert isinstance(messages[0], SystemMessage)
        assert "Codur Code Explainer Agent" in messages[0].content
