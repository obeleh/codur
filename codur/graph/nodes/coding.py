"""Dedicated coding node for the codur-coding agent."""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import ExecuteNodeResult
from codur.graph.nodes.utils import normalize_messages
from codur.graph.nodes.planning.json_parser import JSONResponseParser
from codur.graph.state_operations import (
    get_iterations,
    get_llm_calls,
    get_messages,
    get_tool_calls,
    is_verbose,
)
from codur.utils.llm_calls import invoke_llm
from codur.utils.llm_helpers import invoke_llm_with_fallback
from codur.utils.tool_helpers import extract_tool_info, validate_tool_call
from codur.graph.nodes.tool_executor import execute_tool_calls, get_tool_names
from codur.tools.registry import list_tool_directory

console = Console()


# Built-in system prompt for the coding agent
CODING_AGENT_SYSTEM_PROMPT = """You are Codur Coding Agent, a specialized coding solver.

Your mission: Solve coding requests with correct, efficient, and robust implementations.

## Key Principles

1. **Understand Requirements**: Carefully read the challenge and any provided context.
2. **Docstring Compliance**: If a docstring lists rules/requirements, enumerate them and ensure each is implemented.
3. **Edge Cases First**: Consider boundary conditions, empty inputs, large inputs, special characters.
4. **Correctness Over Cleverness**: Prioritize working code over premature optimization.
5. **Targeted Changes**: Prefer modifying specific functions/classes over rewriting the whole file when possible.
6. **Test File Safety**: Do not overwrite existing test files unless the user is asking to write/update tests there.

## Output Format

You MUST return a valid JSON object with the following structure:

{
  "thought": "Your reasoning about the problem and your plan...",
  "tool_calls": [
    {
      "tool": "replace_function",
      "args": {
        "path": "path/to/file.py",
        "function_name": "name_of_function",
        "new_code": "def name_of_function(...):\\n    ..."
      }
    }
  ]
}

Respond with ONLY a valid JSON object. Do not wrap it in backticks or add extra text.

Examples:
{
  "thought": "Replace title_case with a correct implementation.",
  "tool_calls": [
    {
      "tool": "replace_function",
      "args": {
        "path": "string_utils.py",
        "function_name": "title_case",
        "new_code": "def title_case(sentence: str) -> str:\\n    words = sentence.split()\\n    if not words:\\n        return \\\"\\\"\\n    result = []\\n    for i, word in enumerate(words):\\n        result.append(word.capitalize())\\n    return \\\" \\\".join(result)"
      }
    }
  ]
}

{
  "thought": "Replace the whole file to match the spec.",
  "tool_calls": [
    {
      "tool": "replace_file_content",
      "args": {
        "path": "app.py",
        "new_code": "def main():\\n    print(\\\"ok\\\")\\n\\nif __name__ == \\\"__main__\\\":\\n    main()"
      }
    }
  ]
}

{tools_section}

**Important**:
- `new_code` must be complete, valid Python code (including indentation).
- You can include multiple tool calls in the list if needed.
- Prefer targeted edits (replace_function, replace_class, replace_method, replace_lines) over replace_file_content for tests.
- Tool names must be used EXACTLY as listed above (e.g., "replace_function", NOT "repo_browser.replace_function" or any other prefix).

You MUST return a valid JSON object!
"""


def _parse_signature_to_args(signature: str) -> dict[str, str]:
    """Parse function signature into a JSON-friendly args dict."""
    # Extract parameters from signature like "(path: str, new_code: str, ...)"
    # Remove the parentheses and return type
    sig = signature.strip()
    if sig.startswith("(") and ")" in sig:
        params_str = sig[1:sig.index(")")]
    else:
        return {}

    args = {}
    if not params_str.strip():
        return args

    # Split by comma, but be careful with nested types like list[str]
    params = []
    current = []
    depth = 0
    for char in params_str + ",":
        if char in "[<":
            depth += 1
        elif char in "]>":
            depth -= 1
        if char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    for param in params:
        if not param:
            continue
        # Parse "name: type = default" or "name: type"
        if ":" in param:
            name, rest = param.split(":", 1)
            name = name.strip()
            type_info = rest.split("=")[0].strip()

            # Check if optional (has default or ends with | None)
            is_optional = "=" in rest or "None" in type_info

            # Simplify type info
            if "|" in type_info:
                type_info = type_info.split("|")[0].strip()
            type_info = type_info.replace("'", "").replace('"', "")

            if is_optional:
                args[name] = f"{type_info} (optional)"
            else:
                args[name] = type_info

    return args


def _build_tools_section() -> str:
    """Build the tools section dynamically from the tool registry in JSON format."""
    tools = list_tool_directory()

    # Filter for code modification tools (most relevant for coding agent)
    code_mod_tools = [
        t for t in tools
        if isinstance(t, dict)
        and "name" in t
        and any(keyword in t["name"] for keyword in ["replace", "inject", "rope"])
    ]

    # Also include other useful tools
    other_tools = [
        t for t in tools
        if isinstance(t, dict)
        and "name" in t
        and t not in code_mod_tools
        and any(keyword in t["name"] for keyword in ["read", "write", "find", "validate"])
    ]

    lines = ["## Available Tools", ""]

    if code_mod_tools:
        lines.append("**Code Modification Tools:**")
        lines.append("")
        for i, tool in enumerate(code_mod_tools[:10], 1):  # Limit to top 10
            name = tool["name"]
            sig = tool.get("signature", "")
            summary = tool.get("summary", "")
            args = _parse_signature_to_args(sig)

            # Build compact single-line JSON
            args_json = ", ".join([f'"{k}": "{v}"' for k, v in args.items()])
            json_example = f'{{"tool": "{name}", "args": {{{args_json}}}}}'

            lines.append(f'{i}. **{name}**: {summary}')
            lines.append(f'   {json_example}')
            lines.append("")

    if other_tools[:3]:  # Limit to top 3 useful tools
        lines.append("**Other Useful Tools:**")
        lines.append("")
        for tool in other_tools[:3]:
            name = tool["name"]
            sig = tool.get("signature", "")
            summary = tool.get("summary", "")
            args = _parse_signature_to_args(sig)

            # Build compact single-line JSON
            args_json = ", ".join([f'"{k}": "{v}"' for k, v in args.items()])
            json_example = f'{{"tool": "{name}", "args": {{{args_json}}}}}'

            lines.append(f'- **{name}**: {summary}')
            lines.append(f'  {json_example}')
            lines.append("")

    return "\n".join(lines) if lines else "## Available Tools\n\nNo tools available."


def _build_system_prompt() -> str:
    """Build the complete system prompt with dynamically discovered tools."""
    tools_section = _build_tools_section()
    print("tools section", tools_section)
    return CODING_AGENT_SYSTEM_PROMPT.replace("{tools_section}", tools_section)


def coding_node(state: AgentState, config: CodurConfig) -> ExecuteNodeResult:
    """Run the codur-coding agent with a structured coding prompt.

    Args:
        state: Current graph state with messages, iterations, etc.
        config: Runtime configuration

    Returns:
        ExecuteNodeResult with agent_outcome
    """
    agent_name = "agent:codur-coding"
    iterations = get_iterations(state)
    verbose = is_verbose(state)

    if verbose:
        console.print(f"[bold blue]Running codur-coding node (iteration {iterations})...[/bold blue]")

    # Build context-aware prompt
    prompt = _build_coding_prompt(get_messages(state), iterations)
    if verbose:
        console.print(f"[dim]Constructed prompt:\n{prompt}[/dim]")

    # Use built-in system prompt with dynamic tools
    system_prompt = _build_system_prompt()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    # Invoke LLM
    if verbose:
        console.log("[bold cyan]Invoking codur-coding LLM (JSON mode)...[/bold cyan]")
    
    llm, response = invoke_llm_with_fallback(
        config,
        messages,
        profile_name=config.llm.default_profile,
        fallback_model=config.agents.preferences.fallback_model,
        json_mode=True,
        temperature=config.llm.generation_temperature,
        invoked_by="coding.primary",
        fallback_invoked_by="coding.fallback",
        state=state,
    )

    result = response.content

    # Apply result and check for validation errors
    error = _apply_coding_result(result, state, config)
    if error:
        console.log("[red]Validation/application error detected.[/red]")
        console.log(f"[red]{error}[/red]")
        
        # Validation or application failed - retry once with explicit error
        if iterations >= 1:
            # Already tried at least once, give up
            return {
                "agent_outcome": {
                    "agent": agent_name,
                    "result": f"Failed after retry: {error}",
                    "status": "error",
                }
            }

        # Retry with error feedback
        retry_message = SystemMessage(
            content=f"Previous attempt failed with error:\n{error}\n\nPlease ensure you return valid JSON and correct code."
        )
        console.log("[yellow]Preparing retry prompt...[/yellow]")
        if verbose:
            console.print(f"[dim]{prompt}[/dim]")
            
        messages_for_retry = [
            SystemMessage(content=system_prompt),
            retry_message,
            HumanMessage(content=prompt),
        ]

        if verbose:
            console.log("[yellow]Retrying due to validation error...[/yellow]")

        response = invoke_llm(
            llm,
            messages_for_retry,
            invoked_by="coding.retry",
            state=state,
            config=config,
        )
        result = response.content

        # Try to apply again
        error = _apply_coding_result(result, state, config)
        if error:
            return {
                "agent_outcome": {
                    "agent": agent_name,
                    "result": f"Failed after retry: {error}",
                    "status": "error",
                }
            }

    if verbose:
        console.print(f"[dim]codur-coding response length: {len(result)} chars[/dim]")
        if iterations > 0:
            console.print(f"[yellow]Retry attempt {iterations}[/yellow]")
        if len(result) < 500:
            console.print(f"[dim]Response preview:\n{result}[/dim]")
        else:
            console.print(f"[dim]Response preview:\n{result[:400]}...[/dim]")

    return {
        "agent_outcome": {
            "agent": agent_name,
            "result": result,
            "status": "success",
        },
        "llm_calls": get_llm_calls(state),
    }


def _build_coding_prompt(raw_messages, iterations: int = 0) -> str:
    """Build context-aware prompt from graph state messages."""
    messages = normalize_messages(raw_messages)

    challenge = None
    verification_errors = []
    previous_attempts = []
    context_parts = []

    for message in messages:
        if isinstance(message, HumanMessage) and challenge is None:
            challenge = message.content
        elif isinstance(message, SystemMessage):
            content = message.content
            if "Verification failed" in content or "=== Expected Output ===" in content:
                verification_errors.append(content)
            else:
                context_parts.append(content)
        elif isinstance(message, AIMessage):
            previous_attempts.append(message.content)
        elif isinstance(message, HumanMessage):
            context_parts.append(message.content)

    if challenge is None:
        challenge = messages[-1].content if messages else "No task provided."

    if verification_errors:
        return _build_retry_prompt(challenge, verification_errors[-1], iterations)
    elif context_parts:
        context_text = "\n\n---\n\n".join(context_parts)
        return f"CODING CHALLENGE:\n{challenge}\n\nADDITIONAL CONTEXT:\n{context_text}"
    else:
        return challenge


def _build_retry_prompt(challenge: str, verification_error: str, iterations: int) -> str:
    """Build focused prompt for retry attempts with error context."""
    prompt_parts = [
        f"CODING CHALLENGE (Retry Attempt {iterations}):",
        challenge,
        "",
        "PREVIOUS ATTEMPT FAILED VERIFICATION:",
        verification_error,
        "",
        "INSTRUCTIONS FOR THIS RETRY:",
    ]

    if iterations <= 2:
        prompt_parts.append(
            "- Carefully compare the Expected vs Actual output above\n"
            "- Identify the specific difference (missing line, wrong format, logic error)\n"
            "- Fix the precise issue using targeted replacement tools if possible"
        )
    else:
        prompt_parts.append(
            "- Multiple attempts have failed. Be very systematic:\n"
            "- Check common bugs: off-by-one, edge cases, string formatting\n"
            "- Review the logic carefully and ensure your JSON response is valid"
        )

    return "\n".join(prompt_parts)


def _apply_coding_result(result: str, state: AgentState, config: CodurConfig) -> Optional[str]:
    """
    Parse JSON result and apply tool calls.
    """
    parser = JSONResponseParser()
    data = parser.parse(result)
    if data is None:
        return "Failed to parse JSON response."

    tool_calls = data.get("tool_calls", [])
    if not tool_calls:
        # Fallback: check if 'code' is directly in the object (legacy/fallback)
        if "code" in data:
            default_path = _get_default_path(state, data)
            if not default_path:
                return "Could not determine target file path for code replacement."
            tool_calls = [{
                "tool": "replace_file_content",
                "args": {"path": default_path, "new_code": data["code"]},
            }]
        else:
            return "No 'tool_calls' found in JSON response."

    if not isinstance(tool_calls, list):
        return "Invalid 'tool_calls' format; expected a list."

    data["tool_calls"] = tool_calls
    code_tools = {
        "replace_function",
        "replace_class",
        "replace_method",
        "replace_file_content",
        "inject_function",
    }
    supported_tools = get_tool_names(state, config)
    common_default_path = _get_default_path(state, data)
    for call in tool_calls:
        error = validate_tool_call(call, supported_tools)
        if error:
            return error
        tool_name, args = extract_tool_info(call)
        call["args"] = args
        if tool_name in code_tools and "path" not in args:
            if not common_default_path:
                return f"Missing 'path' for tool '{tool_name}' and could not determine a default."
            args["path"] = common_default_path

    execution = execute_tool_calls(tool_calls, state, config, augment=False, summary_mode="full")
    errors = list(execution.errors)
    for item in execution.results:
        tool_name = item.get("tool")
        output = item.get("output")
        if isinstance(output, str):
            if tool_name in code_tools and _looks_like_tool_failure(output):
                errors.append(output)
            elif tool_name in code_tools:
                console.log(f"[green]{output}[/green]")

    if errors:
        return "\n".join(errors)

    return None


def _looks_like_tool_failure(output: str) -> bool:
    return "Failed" in output or "Invalid" in output or "Could not find" in output


def _get_default_path(state: AgentState, data: dict) -> Optional[str]:
    """Try to determine the target file path from state or JSON data."""
    # 1. Check if any tool call in the current response has a path
    tool_calls = data.get("tool_calls", [])
    for call in tool_calls:
        path = call.get("args", {}).get("path")
        if path:
            return path

    # 2. Check if the state has previous tool calls with a path (from planner)
    planner_tool_calls = get_tool_calls(state)
    for call in planner_tool_calls:
        args = call.get("args", {})
        path = args.get("path") or args.get("file_path")
        if path:
            return path

    return None
