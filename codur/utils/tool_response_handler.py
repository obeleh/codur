"""Handle tool calls from both native API responses and JSON text fallback."""

from langchain_core.messages import AIMessage
from codur.utils.json_parser import JSONResponseParser


def deserialize_tool_calls(message: AIMessage) -> dict:
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
    content = getattr(message, "content")
    if not hasattr(message, 'tool_calls') or not message.tool_calls:
        return {"content": content, "tool_calls": []}

    internal_format = []
    for tool_call in message.tool_calls:
        dct = {
            "tool": tool_call["name"],
            "args": tool_call.get("args", {}),
        }
        if "id" in tool_call:
            dct["id"] = tool_call["id"]
        internal_format.append(dct)

    return {"content": content, "tool_calls": internal_format}


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
