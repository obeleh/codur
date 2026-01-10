"""Dedicated coding node for the codur-coding agent."""
from langchain_core.messages import SystemMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.message_summary import prepend_summary
from codur.graph.state import AgentState
from codur.graph.node_types import ExecuteNodeResult, AgentOutcome
from codur.graph.state_operations import (
    get_iterations,
    get_llm_calls,
    is_verbose,
    increment_iterations,
    add_messages,
    get_next_step_suggestion,
    get_last_tool_output_from_messages,
)
from codur.tools.schema_generator import get_function_schemas
from codur.tools.registry import list_tools_for_tasks
from codur.utils.llm_calls import LLMCallLimitExceeded
from codur.utils.llm_helpers import create_and_invoke_with_tool_support
from codur.constants import TaskType

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
    # Get tools relevant to coding tasks (include unannotated tools as fallback)
    coding_task_types = [
        TaskType.CODE_FIX,
        TaskType.CODE_GENERATION,
        TaskType.CODE_VALIDATION,
        TaskType.FILE_OPERATION,
        TaskType.REFACTOR,
        TaskType.META_TOOL,
    ]
    tools = list_tools_for_tasks(coding_task_types, include_unannotated=True)

    # Categorize tools by TaskType for readability
    file_operation_tools = []
    code_analysis_tools = []
    validation_tools = []
    meta_tools = []
    other_tools = []

    for tool in tools:
        name = tool['name']
        scenarios = tool.get('scenarios', [])

        if TaskType.META_TOOL in scenarios:
            meta_tools.append(name)
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

You have access to the following tools. (only use them if needed):

**File Operations**: {', '.join(sorted(file_operation_tools))}
**Code Analysis & Modification**: {', '.join(sorted(code_analysis_tools))}
**Validation & Testing**: {', '.join(sorted(validation_tools))}
**Other Tools**: {', '.join(sorted(other_tools)) if other_tools else 'None'}
**Meta Tools**: {', '.join(sorted(meta_tools))}
  Meta tools are useful to communicate with the agent framework itself.
  If you are done you can call the done tool

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
    - Make fixes based on validation/execution results (if any)
- If the output of the code was not as expected, include a "clarify" tool call that contains next steps for improvement
- But if the code worked as expected, use the "done" tool to finish. Or use "build_verification_response" if you feel this should be the last step before done.
- When retrying after a failed verification, focus on fixing the specific issues mentioned
"""

# Initialize system prompt with tools
CODING_AGENT_SYSTEM_PROMPT = _get_system_prompt_with_tools()


@prepend_summary
def coding_node(state: AgentState, config: CodurConfig, summary: str, recursion_depth=0) -> ExecuteNodeResult:
    """Run the codur-coding agent with a structured coding prompt.

    Args:
        state: Current graph state with messages, iterations, etc.
        config: Runtime configuration
        summary: Summary of previous messages (injected by decorator)
        recursion_depth: Current recursion depth for nested invocations

    Returns:
        ExecuteNodeResult with agent_outcome
    """
    agent_name = "coding"
    iterations = get_iterations(state)
    increment_iterations(state)
    verbose = is_verbose(state)

    if verbose:
        console.print(f"[bold blue]Running codur-coding node (iteration {iterations})...[/bold blue]")

    tool_schemas = get_function_schemas()  # All 70+ tools

    if recursion_depth == 0:
        suggestion = get_next_step_suggestion(state)
        if suggestion:
            if verbose:
                console.print(f"[dim]Incorporating next step suggestion into prompt:[/dim] {suggestion}")
            summary += f"\n\nNext Step Suggestion: {suggestion}"
        new_messages = [
            SystemMessage(content=CODING_AGENT_SYSTEM_PROMPT + "\n\n" + summary),
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
        if isinstance(exc, LLMCallLimitExceeded):
            raise
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
        return ExecuteNodeResult(
            agent_outcomes=[AgentOutcome(
                agent=agent_name,
                status="success",
                messages=new_messages,
            )],
            messages=new_messages,
            llm_calls=get_llm_calls(state),
        )

    # Meta tool handling
    # Get Last tool to detect "done"
    last_tool_call = get_last_tool_output_from_messages(new_messages)
    if last_tool_call.tool == "done":
        return ExecuteNodeResult(
            agent_outcomes=[AgentOutcome(
                agent=agent_name,
                result=last_tool_call.args["reasoning"],
                status="success",
                messages=new_messages,
            )],
            llm_calls=get_llm_calls(state),
            messages=new_messages,
            selected_agent="codur-verification",
        )
    elif last_tool_call.tool == "build_verification_response":
        console.log("[green]Building verification response as requested by agent...[/green]")
        return ExecuteNodeResult(
            agent_outcomes=[AgentOutcome(
                agent=agent_name,
                result="verification response built",
                status="success",
                messages=new_messages,
            )],
            llm_calls=get_llm_calls(state),
            messages=new_messages,
        )


    if recursion_depth <= 3:
        # inject messages into state for next iteration
        # unfortunately mutations to state do not persist across agent nodes
        add_messages(state, new_messages)
        nested_outcome = coding_node(
            state=state,
            config=config,
            summary=summary,
            recursion_depth=recursion_depth + 1
        )
        nested_outcome["messages"] = new_messages + nested_outcome["messages"]
        return nested_outcome

    return ExecuteNodeResult(
        agent_outcomes=[AgentOutcome(
            agent=agent_name,
            result="tool calls executed",
            status="success",
            messages=new_messages,
        )],
        llm_calls=get_llm_calls(state),
        messages=new_messages,
    )
