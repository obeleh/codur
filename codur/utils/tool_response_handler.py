"""Handle tool calls from both native API responses and JSON text fallback."""

from langchain_core.messages import AIMessage, ToolMessage
from codur.graph.planning.json_parser import JSONResponseParser


def extract_tool_calls_from_ai_message(message: AIMessage) -> list[dict]:
    """
    Convert AIMessage.tool_calls to internal format.

    Input (native API):
        AIMessage(
            content="I'll read that file for you",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "main.py", "line_start": 1, "line_end": 50},
                    "id": "call_abc123"
                }
            ]
        )

    Output (internal format):
        [
            {
                "tool": "read_file",
                "args": {"path": "main.py", "line_start": 1, "line_end": 50},
                "id": "call_abc123"
            }
        ]

    Args:
        message: AIMessage with tool_calls attribute

    Returns:
        List of tool calls in internal format ({"tool": name, "args": dict, "id": str})
    """
    if not hasattr(message, 'tool_calls') or not message.tool_calls:
        return []

    internal_format = []
    for tool_call in message.tool_calls:
        internal_format.append({
            "tool": tool_call["name"],
            "args": tool_call.get("args", {}),
            "id": tool_call.get("id", ""),
        })

    return internal_format


def extract_tool_calls_from_json_text(message: AIMessage) -> list[dict]:
    """
    Extract tool calls from JSON text content (fallback for Ollama).

    Input:
        AIMessage(content='{"thought": "...", "tool_calls": [...]}')

    Output:
        [{"tool": "read_file", "args": {...}}]

    Args:
        message: AIMessage with JSON content

    Returns:
        List of tool calls in internal format
    """
    parser = JSONResponseParser()
    data = parser.parse(message.content)

    if data is None:
        return []

    tool_calls = data.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []

    return tool_calls


def extract_tool_calls_unified(message: AIMessage, used_native: bool) -> list[dict]:
    """
    Extract tool calls from either native API or JSON text fallback.

    Args:
        message: AIMessage from LLM
        used_native: True if native tool calling was used, False for JSON fallback

    Returns:
        List of tool calls in internal format

    Example usage:
        response, used_native = create_and_invoke_with_tool_support(config, messages, schemas)
        tool_calls = extract_tool_calls_unified(response, used_native)
        if tool_calls:
            execute_tool_calls(tool_calls, state, config)
    """
    if used_native:
        return extract_tool_calls_from_ai_message(message)
    else:
        return extract_tool_calls_from_json_text(message)


def create_tool_messages(results: list[dict], tool_call_ids: list[str]) -> list[ToolMessage]:
    """
    Create ToolMessage objects for conversation history.

    Args:
        results: List of tool execution results [{"tool": "read_file", "output": "..."}]
        tool_call_ids: List of tool call IDs from native API

    Returns:
        List of ToolMessage objects for appending to conversation

    Example:
        # After executing tools
        tool_messages = create_tool_messages(execution.results, tool_call_ids)
        messages.extend(tool_messages)
        # Now LLM can see tool results in conversation
    """
    tool_messages = []

    for i, result in enumerate(results):
        tool_name = result.get("tool", "unknown")
        output = result.get("output", "")
        tool_id = tool_call_ids[i] if i < len(tool_call_ids) else ""

        tool_messages.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_id,
                name=tool_name,
            )
        )

    return tool_messages
