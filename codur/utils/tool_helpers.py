"""Utilities for tool call handling and validation."""

from __future__ import annotations

from typing import Any, Iterable


class ToolCallError(ValueError):
    """Error in tool call structure or validation."""


def extract_tool_info(call: dict) -> tuple[str, dict]:
    """Extract and validate tool name and args from a tool call."""
    if not isinstance(call, dict):
        raise ToolCallError(f"Tool call must be a dictionary, got {type(call)}")
    tool_name = call.get("tool")
    if not tool_name:
        raise ToolCallError("Tool call missing 'tool' field")
    if not isinstance(tool_name, str):
        raise ToolCallError(f"Tool name must be string, got {type(tool_name)}")
    args = call.get("args", {})
    if not isinstance(args, dict):
        raise ToolCallError(f"Tool args must be dictionary, got {type(args)}")
    return tool_name, args
