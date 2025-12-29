"""JSON Schema generation from Python function signatures for API tool calling."""

import inspect
from typing import Any, get_args, get_origin, get_type_hints
from codur.tools.registry import list_tool_directory, get_tool_by_name


# Internal parameters to filter out (not exposed to API)
INTERNAL_PARAMS = {"root", "state", "config", "allow_outside_root"}


def _python_type_to_json_type(py_type: Any) -> str:
    """Map Python type to JSON Schema type."""
    # Handle None/NoneType
    if py_type is type(None) or py_type is None:
        return "null"

    # Get origin for generic types (list[str] → list)
    origin = get_origin(py_type)
    if origin is not None:
        py_type = origin

    # Type mapping
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    return type_map.get(py_type, "string")  # Default to string


def _is_optional(annotation: Any) -> bool:
    """Check if type annotation is Optional[T] (Union[T, None])."""
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        # Optional[T] is Union[T, None]
        return type(None) in args
    return False


def _extract_param_description(func: callable, param_name: str) -> str:
    """Extract parameter description from function docstring."""
    docstring = inspect.getdoc(func)
    if not docstring:
        return ""

    # Look for "Args:" section and extract param description
    # Simple parser: look for "param_name: description"
    lines = docstring.split("\n")
    in_args_section = False
    for i, line in enumerate(lines):
        if "Args:" in line or "Parameters:" in line:
            in_args_section = True
            continue
        if in_args_section:
            # End of Args section
            if line.strip() and not line.startswith(" "):
                break
            # Match "param_name: description" or "param_name (type): description"
            if param_name in line and ":" in line:
                # Extract description after colon
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()

    return ""


def function_to_json_schema(func: callable) -> dict:
    """
    Convert Python function to JSON Schema for LangChain tool binding.

    Example input:
        def read_file(path: str, line_start: int = 1, line_end: Optional[int] = None,
                     state: AgentState = None, config: CodurConfig = None) -> str:
            '''Read file contents.

            Args:
                path: File path to read
                line_start: Starting line number
                line_end: Ending line number (optional)
                state: Agent state (internal)
                config: Configuration (internal)
            '''
            ...

    Example output:
        {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    },
                    "line_start": {
                        "type": "integer",
                        "description": "Starting line number"
                    },
                    "line_end": {
                        "type": "integer",
                        "description": "Ending line number (optional)"
                    }
                },
                "required": ["path"]
            }
        }

    Note: Filters out internal params (state, config, root, allow_outside_root).
    """
    sig = inspect.signature(func)

    # Get type hints (handles stringified annotations from __future__ import annotations)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        # Fallback to raw annotations if get_type_hints fails
        type_hints = {}

    # Extract function name and description
    name = func.__name__
    docstring = inspect.getdoc(func) or ""
    # Use first line of docstring as description
    description = docstring.split("\n")[0] if docstring else f"Execute {name}"

    # Build properties and required list
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip internal parameters
        if param_name in INTERNAL_PARAMS:
            continue

        # Get type annotation (prefer type_hints over raw annotation)
        annotation = type_hints.get(param_name, param.annotation)
        if annotation == inspect.Parameter.empty:
            # No type hint, default to string
            param_type = "string"
            is_optional = param.default != inspect.Parameter.empty
        else:
            is_optional = _is_optional(annotation)
            # Extract base type from Optional[T]
            if is_optional:
                args = get_args(annotation)
                # Get first non-None type
                annotation = next((a for a in args if a is not type(None)), str)
            param_type = _python_type_to_json_type(annotation)

        # Extract description from docstring
        param_desc = _extract_param_description(func, param_name)

        # Build property schema (allow null for Optional)
        json_type: str | list[str]
        if is_optional:
            json_type = [param_type, "null"]
        else:
            json_type = param_type
        properties[param_name] = {
            "type": json_type,
            "description": param_desc or f"Parameter {param_name}"
        }

        # Add to required if no default value and not optional
        if param.default == inspect.Parameter.empty and not is_optional:
            required.append(param_name)

    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }


def get_function_schemas(tool_names: list[str] | None = None) -> list[dict]:
    """
    Get JSON schemas for all tools or specific subset.

    Args:
        tool_names: Optional list of tool names to include. If None, returns all tools.

    Returns:
        List of JSON schemas compatible with LangChain's bind_tools()

    Example:
        # Get all tools
        schemas = get_function_schemas()

        # Get specific tools only
        schemas = get_function_schemas(["read_file", "replace_function", "write_file"])
    """
    tools = list_tool_directory()
    schemas = []

    for tool in tools:
        if not isinstance(tool, dict) or "name" not in tool:
            continue

        tool_name = tool["name"]

        # Filter by tool_names if provided
        if tool_names is not None and tool_name not in tool_names:
            continue

        # Get the actual function from registry
        func = get_tool_by_name(tool_name)
        if func is None:
            continue

        # Generate schema
        schema = function_to_json_schema(func)
        schemas.append(schema)

    return schemas
