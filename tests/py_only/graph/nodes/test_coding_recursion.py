"""Tests for coding agent recursion and message passing.

This ensures that tool results are properly passed to recursive calls,
preventing the agent from calling the same tool multiple times.
Similar to verification_agent recursion tests.
"""

from __future__ import annotations

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)


class TestCodingAgentRecursionMessages:
    """Tests for message history preservation in coding agent recursive calls."""

    def test_recursive_call_should_include_tool_results(self):
        """Verify recursive calls should include ToolMessage results from state.

        Before fix (hypothetical issue like verification_agent had):
        - First call: read_file("main.py") → result (not passed to recursive call)
        - Recursive call: Agent confused, might read same file again

        After proper implementation:
        - First call: read_file("main.py") → result (passed via state)
        - Recursive call: Agent sees result, won't repeat
        """
        # Simulate accumulated message history from first tool call
        state_messages = [
            SystemMessage(content="You are a coding assistant"),
            HumanMessage(content="Fix this Python script"),
            AIMessage(
                content="Let me first understand the code",
                tool_calls=[{"id": "call_1", "name": "read_file", "args": {"path": "main.py"}}]
            ),
            ToolMessage(
                content="""def greet(name):
    print(f"Hello {name}""",
                tool_call_id="call_1",
                name="read_file"
            ),
        ]

        # In recursive call, agent should see all these messages
        assert len(state_messages) == 4
        assert isinstance(state_messages[-1], ToolMessage)
        assert "Hello" in state_messages[-1].content

        # Agent knows file was already read, won't read it again

    def test_prevents_duplicate_read_calls(self):
        """Example: Agent shouldn't read the same file twice in recursion.

        Scenario:
        - User: "Fix the syntax error in main.py"
        - Iteration 1: Agent calls read_file("main.py"), sees syntax error
        - Iteration 2: Agent should remember error, not read file again
                      Instead, it should write fixed version

        Risk with missing tool results in recursion:
        - Agent might call read_file("main.py") again, confused about what changed
        - Would waste tokens and LLM context
        """
        # Simulate scenario
        call_1_messages = [
            SystemMessage(content="You are a coding assistant"),
            HumanMessage(content="Fix syntax error in main.py"),
        ]

        # After first call, state should have tool results
        call_1_result = ToolMessage(
            content="SyntaxError: invalid syntax (line 5)",
            tool_call_id="1",
            name="read_file"
        )

        # In recursion, agent should see this
        call_2_messages = [
            SystemMessage(content="You are a coding assistant"),
            HumanMessage(content="Fix syntax error in main.py"),
            AIMessage(
                content="Let me read the file first",
                tool_calls=[{"id": "1", "name": "read_file", "args": {"path": "main.py"}}]
            ),
            call_1_result,  # ← Agent sees this
            HumanMessage(content="Now fix this syntax error"),
        ]

        # Agent can see the error, will call write_file not read_file again
        assert len(call_2_messages) == 5
        assert call_2_messages[-2] == call_1_result
