"""Dedicated coding node for the codur-coding agent."""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import ExecuteNodeResult
from codur.graph.nodes.utils import normalize_messages, resolve_llm_for_model
from codur.utils.llm_calls import invoke_llm
from codur.tools.code_modification import (
    replace_function,
    replace_class,
    replace_method,
    replace_file_content,
    inject_function,
)

console = Console()


# Built-in system prompt for the coding agent
CODING_AGENT_SYSTEM_PROMPT = """You are Codur Coding Agent, a specialized coding solver.

Your mission: Solve coding requests with correct, efficient, and robust implementations.

## Key Principles

1. **Understand Requirements**: Carefully read the challenge and any provided context.
2. **Edge Cases First**: Consider boundary conditions, empty inputs, large inputs, special characters.
3. **Correctness Over Cleverness**: Prioritize working code over premature optimization.
4. **Targeted Changes**: Prefer modifying specific functions/classes over rewriting the whole file when possible.

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

## Available Tools

1. `replace_function(path, function_name, new_code)`: Replace a specific function.
2. `replace_class(path, class_name, new_code)`: Replace a specific class.
3. `replace_method(path, class_name, method_name, new_code)`: Replace a specific method in a class.
4. `replace_file_content(path, new_code)`: Replace the entire file content (use if others don't apply).
5. `inject_function(path, new_code, function_name?)`: Insert a new top-level function (optional name check).

**Important**:
- `new_code` must be complete, valid Python code (including indentation).
- `path` should match the file path in the request
- You can include multiple tool calls in the list if needed.

You MUST return a valid JSON object!
"""


def coding_node(state: AgentState, config: CodurConfig) -> ExecuteNodeResult:
    """Run the codur-coding agent with a structured coding prompt.

    Args:
        state: Current graph state with messages, iterations, etc.
        config: Runtime configuration

    Returns:
        ExecuteNodeResult with agent_outcome
    """
    agent_name = "agent:codur-coding"
    iterations = state.get("iterations", 0)
    verbose = state.get("verbose", False)

    if verbose:
        console.print(f"[bold blue]Running codur-coding node (iteration {iterations})...[/bold blue]")

    # Resolve LLM (uses default LLM - system prompt is self-contained)
    # Use generation temperature for coding tasks
    # Enable JSON mode for structured output
    llm = resolve_llm_for_model(
        config,
        None,
        temperature=config.llm.generation_temperature,
        json_mode=True
    )

    # Build context-aware prompt
    prompt = _build_coding_prompt(state.get("messages", []), iterations)
    if verbose:
        console.print(f"[dim]Constructed prompt:\n{prompt}[/dim]")

    # Use built-in system prompt
    messages = [
        SystemMessage(content=CODING_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Invoke LLM
    if verbose:
        console.log("[bold cyan]Invoking codur-coding LLM (JSON mode)...[/bold cyan]")
    
    try:
        response = invoke_llm(
            llm,
            messages,
            invoked_by="coding.primary",
            state=state,
            config=config,
        )
    except Exception as e:
        # Check for JSON validation error (common with some Groq models like Qwen)
        error_msg = str(e)
        if "json_validate_failed" in error_msg or ("400" in error_msg and "JSON" in error_msg):
            fallback_model = config.agents.preferences.fallback_model
            console.log(f"[yellow]Primary LLM failed JSON validation. Falling back to {fallback_model}...[/yellow]")
            fallback_llm = resolve_llm_for_model(
                config, 
                model=fallback_model,
                temperature=config.llm.generation_temperature,
                json_mode=True
            )
            response = invoke_llm(
                fallback_llm,
                messages,
                invoked_by="coding.fallback",
                state=state,
                config=config,
            )
        else:
            raise e

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
            SystemMessage(content=CODING_AGENT_SYSTEM_PROMPT),
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
        "llm_calls": state.get("llm_calls", 0),
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
    try:
        # Simple JSON extraction if wrapped in code blocks
        if "```json" in result:
            json_str = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            json_str = result.split("```")[1].split("```")[0].strip()
        else:
            json_str = result

        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"Failed to parse JSON response: {str(e)}"

    tool_calls = data.get("tool_calls", [])
    if not tool_calls:
        # Fallback: check if 'code' is directly in the object (legacy/fallback)
        if "code" in data:
             # Treat as full file replacement
             default_path = _get_default_path(state, data)
             if not default_path:
                 return "Could not determine target file path for code replacement."
             return replace_file_content(path=default_path, new_code=data["code"], root=Path.cwd(), allow_outside_root=config.runtime.allow_outside_workspace, state=state)
        return "No 'tool_calls' found in JSON response."

    errors = []
    
    # Map tool names to functions
    tool_map = {
        "replace_function": replace_function,
        "replace_class": replace_class,
        "replace_method": replace_method,
        "replace_file_content": replace_file_content,
        "inject_function": inject_function,
    }

    # Pre-determine a default path for all tool calls in this result
    common_default_path = _get_default_path(state, data)

    for call in tool_calls:
        tool_name = call.get("tool")
        args = call.get("args", {})
        
        if tool_name not in tool_map:
            errors.append(f"Unknown tool: {tool_name}")
            continue
            
        # Execute tool
        try:
            func = tool_map[tool_name]
            # Inject common args
            args["root"] = Path.cwd()
            args["allow_outside_root"] = config.runtime.allow_outside_workspace
            args["state"] = state
            
            # Use dynamic default path if missing
            if "path" not in args:
                if not common_default_path:
                    errors.append(f"Missing 'path' for tool '{tool_name}' and could not determine a default.")
                    continue
                args["path"] = common_default_path
                
            outcome = func(**args)
            if "Failed" in outcome or "Invalid" in outcome or "Could not find" in outcome:
                errors.append(outcome)
            else:
                console.log(f"[green]{outcome}[/green]")
                
        except Exception as e:
            errors.append(f"Error executing {tool_name}: {str(e)}")

    if errors:
        return "\n".join(errors)

    return None


def _get_default_path(state: AgentState, data: dict) -> Optional[str]:
    """Try to determine the target file path from state or JSON data."""
    # 1. Check if any tool call in the current response has a path
    tool_calls = data.get("tool_calls", [])
    for call in tool_calls:
        path = call.get("args", {}).get("path")
        if path:
            return path

    # 2. Check if the state has previous tool calls with a path (from planner)
    planner_tool_calls = state.get("tool_calls", [])
    for call in planner_tool_calls:
        args = call.get("args", {})
        path = args.get("path") or args.get("file_path")
        if path:
            return path

    return None
