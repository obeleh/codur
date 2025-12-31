"""Tests for message_shortening_pipeline function."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from codur.utils.custom_messages import ShortenableSystemMessage
from codur.utils.llm_helpers import message_shortening_pipeline


# Mock tools with side effects for consistent test behavior
MOCK_MUTATION_TOOLS = {"write_file", "delete_file", "git_commit"}


@pytest.fixture(autouse=True)
def mock_tools_with_side_effects():
    """Mock get_tools_with_side_effects to return predictable tool names."""
    with patch(
        "codur.utils.llm_helpers.get_tools_with_side_effects",
        return_value=list(MOCK_MUTATION_TOOLS),
    ):
        yield


def make_tool_message(tool_name: str, content: str = "result") -> ToolMessage:
    """Helper to create a ToolMessage with required fields."""
    return ToolMessage(content=content, tool_call_id="test-id", name=tool_name)


class TestNonMutationToolCallsBeforeShortenable:
    """Non-mutation tool calls before first ShortenableMessage."""

    def test_non_mutation_tools_before_shortenable_are_preserved(self):
        """Non-mutation tool calls before ShortenableSystemMessage are preserved."""
        messages = [
            make_tool_message("read_file", "file content"),
            make_tool_message("list_files", "files list"),
            ShortenableSystemMessage(content="system context"),
            HumanMessage(content="do something"),
        ]

        result = message_shortening_pipeline(messages)

        # Without mutation, all tool calls are injected after shortenable
        assert len(result) == 4
        assert isinstance(result[0], ShortenableSystemMessage)
        assert result[1].content == "file content"
        assert result[2].content == "files list"
        assert isinstance(result[3], HumanMessage)

    def test_multiple_non_mutation_tools_before_shortenable(self):
        """Multiple non-mutation tools are all preserved."""
        messages = [
            make_tool_message("read_file"),
            make_tool_message("search_code"),
            make_tool_message("list_files"),
            make_tool_message("get_tree"),
            ShortenableSystemMessage(content="context"),
        ]

        result = message_shortening_pipeline(messages)

        # All 4 tool calls + shortenable
        assert len(result) == 5
        assert isinstance(result[0], ShortenableSystemMessage)
        assert result[1].name == "read_file"
        assert result[2].name == "search_code"
        assert result[3].name == "list_files"
        assert result[4].name == "get_tree"


class TestMutationToolCallInMiddle:
    """Mutation tool call in the middle of tool calls."""

    def test_mutation_tool_preserves_subsequent_tool_calls(self):
        """Tool calls after a mutation tool are preserved."""
        messages = [
            make_tool_message("read_file"),
            make_tool_message("write_file"),  # mutation
            make_tool_message("read_file", "verify result"),
            ShortenableSystemMessage(content="context"),
            HumanMessage(content="continue"),
        ]

        result = message_shortening_pipeline(messages)

        # ShortenableSystemMessage + tool call after mutation + HumanMessage
        assert len(result) == 3
        assert isinstance(result[0], ShortenableSystemMessage)
        assert isinstance(result[1], ToolMessage)
        assert result[1].name == "read_file"
        assert result[1].content == "verify result"
        assert isinstance(result[2], HumanMessage)

    def test_mutation_in_middle_drops_before_preserves_after(self):
        """Tools before mutation are dropped, tools after are preserved."""
        messages = [
            make_tool_message("read_file", "before1"),
            make_tool_message("list_files", "before2"),
            make_tool_message("delete_file"),  # mutation
            make_tool_message("read_file", "after1"),
            make_tool_message("search_code", "after2"),
            ShortenableSystemMessage(content="context"),
        ]

        result = message_shortening_pipeline(messages)

        # ShortenableSystemMessage + 2 tool calls after mutation
        assert len(result) == 3
        assert isinstance(result[0], ShortenableSystemMessage)
        assert result[1].content == "after1"
        assert result[2].content == "after2"


class TestMultipleShortenableMessages:
    """Multiple ShortenableMessages with tool calls in between."""

    def test_only_last_shortenable_is_kept(self):
        """Only the last ShortenableSystemMessage is preserved, with tool calls injected."""
        messages = [
            ShortenableSystemMessage(content="first context"),
            HumanMessage(content="task 1"),
            make_tool_message("read_file"),
            ShortenableSystemMessage(content="second context"),
            HumanMessage(content="task 2"),
            make_tool_message("list_files"),
            ShortenableSystemMessage(content="third context"),
            HumanMessage(content="task 3"),
        ]

        result = message_shortening_pipeline(messages)

        # Last ShortenableSystemMessage + all tool calls before it + HumanMessage after it
        assert len(result) == 4
        assert isinstance(result[0], ShortenableSystemMessage)
        assert result[0].content == "third context"
        assert result[1].name == "read_file"  # tool call injected
        assert result[2].name == "list_files"  # tool call injected
        assert isinstance(result[3], HumanMessage)
        assert result[3].content == "task 3"

    def test_tool_calls_between_shortenables_without_mutation(self):
        """Tool calls between ShortenableMessages without mutation are preserved."""
        messages = [
            ShortenableSystemMessage(content="ctx1"),
            make_tool_message("read_file", "read1"),
            ShortenableSystemMessage(content="ctx2"),
            make_tool_message("list_files", "list1"),
            ShortenableSystemMessage(content="ctx3"),
        ]

        result = message_shortening_pipeline(messages)

        # All tool calls before last shortenable are injected
        assert len(result) == 3
        assert result[0].content == "ctx3"
        assert result[1].content == "read1"
        assert result[2].content == "list1"


class TestMultipleShortenablesWithMutation:
    """Multiple ShortenableMessages with mutating tool calls in between."""

    def test_mutation_before_last_shortenable_preserves_subsequent_tools(self):
        """Mutation before last ShortenableSystemMessage preserves tools after it."""
        messages = [
            ShortenableSystemMessage(content="ctx1"),
            make_tool_message("write_file"),  # mutation
            make_tool_message("read_file", "verify1"),
            ShortenableSystemMessage(content="ctx2"),
            HumanMessage(content="next"),
        ]

        result = message_shortening_pipeline(messages)

        # Last shortenable + tool after mutation + HumanMessage
        assert len(result) == 3
        assert result[0].content == "ctx2"
        assert result[1].name == "read_file"
        assert result[2].content == "next"

    def test_multiple_mutations_only_last_matters(self):
        """Only tools after the last mutation are preserved."""
        messages = [
            ShortenableSystemMessage(content="ctx1"),
            make_tool_message("write_file"),  # first mutation
            make_tool_message("read_file", "after_first"),
            make_tool_message("git_commit"),  # second mutation
            make_tool_message("read_file", "after_second"),
            ShortenableSystemMessage(content="ctx2"),
        ]

        result = message_shortening_pipeline(messages)

        # Last shortenable + only tool after last mutation
        assert len(result) == 2
        assert result[0].content == "ctx2"
        assert result[1].content == "after_second"

    def test_mutation_after_last_shortenable(self):
        """Mutation after last ShortenableSystemMessage - all prior tools preserved.

        When mutation is after the last shortenable, all tool calls from 0 to
        shortenable are preserved (since there's no mutation before shortenable).
        """
        messages = [
            ShortenableSystemMessage(content="ctx1"),
            make_tool_message("read_file", "before"),
            ShortenableSystemMessage(content="ctx2"),
            make_tool_message("write_file"),  # mutation after last shortenable
            make_tool_message("read_file", "verify"),
            HumanMessage(content="done"),
        ]

        result = message_shortening_pipeline(messages)

        # All tool calls before shortenable are preserved + messages after shortenable
        assert len(result) == 5
        assert result[0].content == "ctx2"
        assert result[1].content == "before"  # tool call before shortenable
        assert result[2].name == "write_file"  # from messages after shortenable
        assert result[3].content == "verify"
        assert result[4].content == "done"


class TestCornerCases:
    """Corner cases and edge scenarios."""

    def test_no_shortenable_raises_error(self):
        """Missing ShortenableSystemMessage raises ValueError."""
        messages = [
            HumanMessage(content="task"),
            make_tool_message("read_file"),
        ]

        with pytest.raises(ValueError, match="No ShortenableSystemMessage found"):
            message_shortening_pipeline(messages)

    def test_empty_messages_raises_error(self):
        """Empty message list raises ValueError."""
        with pytest.raises(ValueError, match="No ShortenableSystemMessage found"):
            message_shortening_pipeline([])

    def test_only_shortenable_message(self):
        """Single ShortenableSystemMessage returns just that message."""
        messages = [ShortenableSystemMessage(content="only")]

        result = message_shortening_pipeline(messages)

        assert len(result) == 1
        assert result[0].content == "only"

    def test_shortenable_at_end(self):
        """ShortenableSystemMessage at end of message list with tool calls preserved."""
        messages = [
            HumanMessage(content="task"),
            AIMessage(content="response"),
            make_tool_message("read_file"),
            ShortenableSystemMessage(content="final context"),
        ]

        result = message_shortening_pipeline(messages)

        # Tool call before shortenable is preserved
        assert len(result) == 2
        assert result[0].content == "final context"
        assert result[1].name == "read_file"

    def test_ai_messages_after_shortenable_preserved(self):
        """Non-tool messages after ShortenableSystemMessage are preserved."""
        messages = [
            make_tool_message("read_file"),
            ShortenableSystemMessage(content="context"),
            AIMessage(content="thinking"),
            HumanMessage(content="feedback"),
            AIMessage(content="more thinking"),
        ]

        result = message_shortening_pipeline(messages)

        # Tool call before shortenable is also preserved
        assert len(result) == 5
        assert isinstance(result[0], ShortenableSystemMessage)
        assert result[1].name == "read_file"  # tool call injected
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[3], HumanMessage)
        assert isinstance(result[4], AIMessage)

    def test_mutation_at_very_end(self):
        """Mutation tool as last message (after shortenable)."""
        messages = [
            ShortenableSystemMessage(content="context"),
            HumanMessage(content="request"),
            make_tool_message("write_file"),
        ]

        result = message_shortening_pipeline(messages)

        # All messages after shortenable are preserved
        assert len(result) == 3
        assert result[0].content == "context"
        assert result[1].content == "request"
        assert result[2].name == "write_file"

    def test_no_mutations_all_tool_calls_preserved(self):
        """Without mutations, all tool calls before shortenable are preserved."""
        messages = [
            make_tool_message("read_file", "r1"),
            make_tool_message("search_code", "s1"),
            ShortenableSystemMessage(content="context"),
            HumanMessage(content="task"),
        ]

        result = message_shortening_pipeline(messages)

        # Without mutation, all tool calls are preserved
        assert len(result) == 4
        assert isinstance(result[0], ShortenableSystemMessage)
        assert result[1].content == "r1"
        assert result[2].content == "s1"
        assert isinstance(result[3], HumanMessage)

    def test_consecutive_mutations(self):
        """Multiple consecutive mutation tools."""
        messages = [
            make_tool_message("write_file"),
            make_tool_message("delete_file"),
            make_tool_message("git_commit"),  # last mutation
            make_tool_message("read_file", "final check"),
            ShortenableSystemMessage(content="context"),
        ]

        result = message_shortening_pipeline(messages)

        # Only tool after last mutation + shortenable
        assert len(result) == 2
        assert result[0].content == "context"
        assert result[1].content == "final check"

    def test_only_for_agent_parameter_accepted(self):
        """The only_for_agent parameter is accepted (currently unused in new impl)."""
        messages = [
            ShortenableSystemMessage(content="context"),
            HumanMessage(content="task"),
        ]

        # Should not raise - parameter is accepted even if not used
        result = message_shortening_pipeline(messages, only_for_agent="coding")

        assert len(result) == 2

    def test_mixed_message_types_complex(self):
        """Complex scenario with various message types."""
        messages = [
            HumanMessage(content="initial"),
            make_tool_message("read_file", "read1"),
            AIMessage(content="thinking"),
            ShortenableSystemMessage(content="ctx1"),
            make_tool_message("write_file"),  # mutation
            make_tool_message("read_file", "verify"),
            HumanMessage(content="continue"),
            AIMessage(content="more thinking"),
            ShortenableSystemMessage(content="ctx2"),
            make_tool_message("list_files", "listing"),
            HumanMessage(content="final"),
        ]

        result = message_shortening_pipeline(messages)

        # Last shortenable + tool after mutation (verify) + messages after shortenable
        assert len(result) == 4
        assert result[0].content == "ctx2"
        assert result[1].content == "verify"  # Tool after mutation
        assert result[2].content == "listing"  # Tool after last shortenable
        assert result[3].content == "final"

    def test_tool_calls_only_after_mutation_before_shortenable(self):
        """Tool calls between mutation and shortenable are preserved."""
        messages = [
            make_tool_message("read_file", "initial"),
            make_tool_message("write_file"),  # mutation
            make_tool_message("read_file", "check1"),
            make_tool_message("list_files", "check2"),
            ShortenableSystemMessage(content="context"),
        ]

        result = message_shortening_pipeline(messages)

        # Shortenable + 2 tool calls after mutation
        assert len(result) == 3
        assert result[0].content == "context"
        assert result[1].content == "check1"
        assert result[2].content == "check2"