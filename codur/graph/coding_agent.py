"""Dedicated coding node for the codur-coding agent."""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.node_types import ExecuteNodeResult
from codur.graph.utils import normalize_messages
from codur.graph.state_operations import (
    get_iterations,
    get_llm_calls,
    get_messages,
    is_verbose, increment_iterations, add_messages
)
from codur.tools.schema_generator import get_function_schemas
from codur.utils.llm_helpers import create_and_invoke_with_tool_support

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
    from codur.tools.registry import list_tools_for_tasks
    from codur.constants import TaskType

    # Get tools relevant to coding tasks (include unannotated tools as fallback)
    coding_task_types = [
        TaskType.CODE_FIX,
        TaskType.CODE_GENERATION,
        TaskType.CODE_VALIDATION,
        TaskType.FILE_OPERATION,
        TaskType.REFACTOR,
    ]
    tools = list_tools_for_tasks(coding_task_types, include_unannotated=True)

    # Categorize tools by TaskType for readability
    file_operation_tools = []
    code_analysis_tools = []
    validation_tools = []
    other_tools = []

    for tool in tools:
        name = tool['name']
        scenarios = tool.get('scenarios', [])

        if TaskType.FILE_OPERATION in scenarios:
            file_operation_tools.append(name)
        elif TaskType.CODE_VALIDATION in scenarios:
            validation_tools.append(name)
        elif TaskType.CODE_FIX in scenarios or TaskType.CODE_GENERATION in scenarios or TaskType.REFACTOR in scenarios:
            code_analysis_tools.append(name)
        else:
            other_tools.append(name)

    tools_section = f"""
## Available Tools

You have access to the following tools. Call them directly - they will be executed automatically.

**File Operations**: {', '.join(sorted(file_operation_tools))}
**Code Analysis & Modification**: {', '.join(sorted(code_analysis_tools))}
**Validation & Testing**: {', '.join(sorted(validation_tools))}
**Other Tools**: {', '.join(sorted(other_tools)) if other_tools else 'None'}

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
- You can read multiple files in one call using read_files, write_files
- All tool arguments must match the schema exactly
- After Writing Code (replace_function, write_file, replace_class, etc.):
    - For Python files: Use validate_python_syntax to verify syntax is correct
    - To test execution: Use run_python_file to run the modified code and see output
    - Validation and execution are faster and more efficient than reading the entire file back
    - Only run pytest if there are tests
"""

# Initialize system prompt with tools
CODING_AGENT_SYSTEM_PROMPT = _get_system_prompt_with_tools()


def coding_node(state: AgentState, config: CodurConfig, recursion_depth=0) -> ExecuteNodeResult:
    """Run the codur-coding agent with a structured coding prompt.

    Args:
        state: Current graph state with messages, iterations, etc.
        config: Runtime configuration

    Returns:
        ExecuteNodeResult with agent_outcome
    """
    agent_name = "agent:codur-coding"
    iterations = get_iterations(state)
    increment_iterations(state)
    verbose = is_verbose(state)

    if verbose:
        console.print(f"[bold blue]Running codur-coding node (iteration {iterations})...[/bold blue]")

    # Build context-aware prompt
    prompt = _build_coding_prompt(get_messages(state), iterations)

    tool_schemas = get_function_schemas()  # All 70+ tools

    if recursion_depth == 0:
        new_messages = [
            SystemMessage(content=CODING_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    else:
        new_messages = []

    try:
        new_messages, execution_result = create_and_invoke_with_tool_support(
            config,
            new_messages,
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
            new_messages, execution_result = create_and_invoke_with_tool_support(
                config,
                new_messages,
                tool_schemas,
                profile_name=fallback_profile,
                temperature=config.llm.generation_temperature,
                invoked_by="coding.fallback",
                state=state,
            )
        else:
            raise

    if not execution_result.results:
        # No tools called - return response as-is
        return {
            "agent_outcome": {
                "agent": agent_name,
                "status": "success",
            },
            "messages": new_messages,
            "llm_calls": get_llm_calls(state),
        }

    if recursion_depth < 3:
        # inject messages into state for next iteration
        # unfortunately mutations to state do not persist across agent nodes
        add_messages(state, new_messages)
        nested_outcome = coding_node(state, config, recursion_depth + 1)
        nested_outcome["messages"] = new_messages + nested_outcome["messages"]
        return nested_outcome

    return {
        "agent_outcome": {
            "agent": agent_name,
            "result": "tool calls executed",
            "status": "success",
        },
        "llm_calls": get_llm_calls(state),
        "messages": new_messages,
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


def _unify_result_message(result: str | list[str]) -> str:
    """Unify result message into a single string."""
    if isinstance(result, list):
        return "\n".join(result)
    return result
