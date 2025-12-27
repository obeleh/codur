"""Dedicated coding node for the codur-coding agent."""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import ExecuteNodeResult
from codur.graph.nodes.utils import normalize_messages
from codur.graph.state_operations import (
    get_iterations,
    get_llm_calls,
    get_messages,
    is_verbose,
)
from codur.tools.schema_generator import get_function_schemas
from codur.utils.llm_helpers import create_and_invoke_with_tool_support
from codur.utils.tool_response_handler import extract_tool_calls_unified
from codur.graph.nodes.tool_executor import execute_tool_calls

"""
NOTE:
    - Tool calls (or error handling thereof) should be executed by execute_tool_calls
    - No local optimizations that should be global ones
    - No code in this file specific to the challenges, the agent should be generally applicable
    - State operations should use codur.graph.state_operations
"""

console = Console()


# Build system prompt with available tools
def _get_system_prompt_with_tools():
    """Build system prompt with available tools listed."""
    from codur.tools.registry import list_tool_directory

    tools = list_tool_directory()
    tool_names = sorted([t['name'] for t in tools])

    # Categorize tools for readability
    file_tools = [n for n in tool_names if any(x in n for x in ['read', 'write', 'file', 'replace', 'inject'])]
    code_tools = [n for n in tool_names if any(x in n for x in ['python_ast', 'function', 'class', 'method', 'rope'])]
    other_tools = [n for n in tool_names if n not in file_tools and n not in code_tools]

    tools_section = f"""
## Available Tools

You have access to the following tools. Call them directly - they will be executed automatically.

**File Operations**: {', '.join(sorted(file_tools)[:15])}{"..." if len(file_tools) > 15 else ""}

**Code Analysis & Modification**: {', '.join(sorted(code_tools)[:20])}{"..." if len(code_tools) > 20 else ""}

**Other**: {', '.join(sorted(other_tools)[:10])}{"..." if len(other_tools) > 10 else ""}

**CRITICAL**: Only use tools from this list. Do NOT invent or create new tools.
All code modification tools (replace_function, write_file, etc.) require a 'path' parameter.
"""

    return f"""You are Codur Coding Agent, a specialized coding solver.

Your mission: Solve coding requests with correct, efficient, and robust implementations.

## Key Principles

1. **Understand Requirements**: Carefully read the challenge and any provided context.
2. **Docstring Compliance**: If a docstring lists rules/requirements, enumerate them and ensure each is implemented.
3. **Edge Cases First**: Consider boundary conditions, empty inputs, large inputs, special characters.
4. **Correctness Over Cleverness**: Prioritize working code over premature optimization.
5. **Targeted Changes**: Prefer modifying specific functions/classes over rewriting the whole file when possible.
6. **Test File Safety**: Do not overwrite existing test files unless the user is asking to write/update tests there.
7. **Use Existing Results**: Check previous messages for tool call results - reuse them rather than re-reading files.
{tools_section}
## Important Notes

- You MUST return valid tool calls - do NOT create fake tool names or prefixes
- If you need to read a file multiple times in the same conversation, use the results from the first read
- All tool arguments must match the schema exactly
- Do NOT attempt to run code locally - use the tools provided
"""

# Initialize system prompt with tools
CODING_AGENT_SYSTEM_PROMPT = _get_system_prompt_with_tools()


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

    tool_schemas = get_function_schemas()  # All 70+ tools

    # Simplified system prompt (no tool injection)
    system_prompt = CODING_AGENT_SYSTEM_PROMPT
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    # Invoke with tool support (native or JSON fallback)
    if verbose:
        console.log("[bold cyan]Invoking codur-coding LLM with native tools...[/bold cyan]")
    try:
        response, used_native = create_and_invoke_with_tool_support(
            config,
            messages,
            tool_schemas,
            profile_name=config.llm.default_profile,
            temperature=config.llm.generation_temperature,
            invoked_by="coding.primary",
            state=state,
        )
    except Exception as exc:
        # Fallback on error
        if verbose:
            console.log(f"[yellow]Primary invocation failed: {exc}[/yellow]")

        # Try fallback model
        fallback_profile = config.agents.preferences.fallback_model
        if fallback_profile:
            response, used_native = create_and_invoke_with_tool_support(
                config,
                messages,
                tool_schemas,
                profile_name=fallback_profile,
                temperature=config.llm.generation_temperature,
                invoked_by="coding.fallback",
                state=state,
            )
        else:
            raise

    # Extract tool calls (works for both native and JSON)
    tool_calls = extract_tool_calls_unified(response, used_native)

    if not tool_calls:
        # No tools called - return response as-is
        return {
            "agent_outcome": {
                "agent": agent_name,
                "result": response.content,
                "status": "success",
            },
            "llm_calls": get_llm_calls(state),
        }

    # Execute tools
    if verbose:
        console.log(f"[cyan]Executing {len(tool_calls)} tool call(s)...[/cyan]")

    execution = execute_tool_calls(tool_calls, state, config, augment=False, summary_mode="full")
    errors = list(execution.errors)

    for item in execution.results:
        tool_name = item.get("tool")
        output = item.get("output")
        if isinstance(output, str):
            if _looks_like_tool_failure(output):
                errors.append(output)

    if errors:
        error_text = "\n".join(errors)

        # Retry once on error
        if iterations >= 1:
            return {
                "agent_outcome": {
                    "agent": agent_name,
                    "result": f"Failed after retry: {error_text}",
                    "status": "error",
                }
            }

        # Retry with error feedback
        retry_message = SystemMessage(
            content=f"Previous attempt failed with error:\n{error_text}\n\nPlease fix and try again."
        )
        messages_for_retry = [
            SystemMessage(content=system_prompt),
            retry_message,
            HumanMessage(content=prompt),
        ]

        if verbose:
            console.log("[yellow]Retrying due to tool execution error...[/yellow]")

        response, used_native = create_and_invoke_with_tool_support(
            config,
            messages_for_retry,
            tool_schemas,
            profile_name=config.llm.default_profile,
            temperature=config.llm.generation_temperature,
            invoked_by="coding.retry",
            state=state,
        )

        # Extract and execute again
        tool_calls = extract_tool_calls_unified(response, used_native)
        if tool_calls:
            execution = execute_tool_calls(tool_calls, state, config, augment=False, summary_mode="full")
            errors = list(execution.errors)
            if errors:
                return {
                    "agent_outcome": {
                        "agent": agent_name,
                        "result": f"Failed after retry: {'\n'.join(errors)}",
                        "status": "error",
                    }
                }

    # Tool execution successful - feed results back to LLM for continuation
    if verbose:
        console.print("[green]Tool execution completed - feeding results back to LLM[/green]")

    # Implement tool loop: keep invoking LLM with tool results until it's done
    max_tool_iterations = 5
    tool_iteration = 1

    # Determine the working file path from initial tool calls
    working_file = None
    for call in tool_calls:
        args = call.get("args", {})
        if "path" in args:
            working_file = args["path"]
            break

    # Add tool results with clear instructions to use them and not re-read
    tool_result_msg = f"Tool results:\n{execution.summary}"

    # Track which files were read to prevent re-reading
    files_read = set()
    for call in tool_calls:
        if call.get("tool") == "read_file":
            path = call.get("args", {}).get("path")
            if path:
                files_read.add(path)

    if working_file:
        tool_result_msg += f"\n\n⚠️ You are working on: {working_file}"
        tool_result_msg += f"\n⚠️ All code modification tools MUST include: path: {working_file}"

    if files_read:
        tool_result_msg += f"\n\n⚠️ Files already read (DO NOT read again): {', '.join(sorted(files_read))}"
        tool_result_msg += f"\n⚠️ Use the content above to implement the solution - do NOT call read_file again!"

    conversation_messages = messages + [
        AIMessage(content=response.content),
        SystemMessage(content=tool_result_msg)
    ]

    while tool_iteration < max_tool_iterations:
        if verbose:
            console.log(f"[cyan]Tool iteration {tool_iteration}: invoking LLM with tool results[/cyan]")

        # Invoke LLM with accumulated conversation including tool results
        try:
            response, used_native = create_and_invoke_with_tool_support(
                config,
                conversation_messages,
                tool_schemas,
                profile_name=config.llm.default_profile,
                temperature=config.llm.generation_temperature,
                invoked_by=f"coding.tool_loop.{tool_iteration}",
                state=state,
            )
        except Exception as e:
            error_msg = str(e)
            if verbose:
                console.log(f"[red]LLM invocation failed in tool loop: {error_msg}[/red]")
            # Return the error with more context
            return {
                "agent_outcome": {
                    "agent": agent_name,
                    "result": f"LLM error in tool loop iteration {tool_iteration}: {error_msg}",
                    "status": "error",
                },
                "llm_calls": get_llm_calls(state),
            }

        # Extract tool calls from response
        tool_calls = extract_tool_calls_unified(response, used_native)

        if not tool_calls:
            # No more tool calls - LLM is done, return final response
            if verbose:
                console.log(f"[green]LLM finished after {tool_iteration} tool iteration(s)[/green]")
            return {
                "agent_outcome": {
                    "agent": agent_name,
                    "result": response.content,
                    "status": "success",
                },
                "llm_calls": get_llm_calls(state),
            }

        # Execute the new tool calls
        if verbose:
            console.log(f"[cyan]Executing {len(tool_calls)} tool call(s)...[/cyan]")

        execution = execute_tool_calls(tool_calls, state, config, augment=False, summary_mode="full")

        # Check for errors
        errors = [item.get("output") for item in execution.results if _looks_like_tool_failure(item.get("output", ""))]
        if errors:
            error_text = "\n".join(errors)
            if verbose:
                console.log(f"[red]Tool execution failed: {error_text}[/red]")
            return {
                "agent_outcome": {
                    "agent": agent_name,
                    "result": f"Tool execution failed: {error_text}",
                    "status": "error",
                },
                "llm_calls": get_llm_calls(state),
            }

        # Add to conversation and continue loop
        conversation_messages.append(AIMessage(content=response.content))
        conversation_messages.append(SystemMessage(content=f"Tool results:\n{execution.summary}"))
        tool_iteration += 1

    # Max iterations reached
    if verbose:
        console.log(f"[yellow]Max tool iterations ({max_tool_iterations}) reached[/yellow]")

    return {
        "agent_outcome": {
            "agent": agent_name,
            "result": response.content,
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


def _looks_like_tool_failure(output: str) -> bool:
    return bool("Failed" in output or "Invalid" in output or "Could not find" in output)

