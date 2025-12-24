"""Tests for the explaining node."""

from unittest.mock import MagicMock, patch

from codur.graph.nodes.explaining import explaining_node, _build_explaining_prompt
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class TestExplainingNode:
    """Test the explaining node."""

    def test_build_explaining_prompt(self):
        """Test prompt construction."""
        messages = [
            HumanMessage(content="Explain main.py"),
            SystemMessage(content="Content of main.py"),
            AIMessage(content="I see."),
        ]
        prompt = _build_explaining_prompt(messages)
        assert "EXPLANATION REQUEST:\nExplain main.py" in prompt
        assert "ADDITIONAL CONTEXT:\nContent of main.py" in prompt
        assert "Previous response: I see." in prompt

    @patch("codur.graph.nodes.explaining.invoke_llm")
    @patch("codur.graph.nodes.explaining._resolve_llm_for_model")
    def test_explaining_node_execution(self, mock_resolve, mock_invoke):
        """Test that the node executes and returns a result."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_resolve.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = "This is the explanation."
        mock_invoke.return_value = mock_response

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
        assert result["agent_outcome"]["agent"] == "agent:codur-explaining"
        assert result["agent_outcome"]["result"] == "This is the explanation."
        assert result["agent_outcome"]["status"] == "success"
        
        # Verify LLM call
        mock_invoke.assert_called_once()
        args, kwargs = mock_invoke.call_args
        messages = args[1]
        assert isinstance(messages[0], SystemMessage)
        assert "Codur Explaining Agent" in messages[0].content
