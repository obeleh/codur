"""Dedicated verification node for the codur-verification agent."""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.message_summary import prepend_summary
from codur.graph.state import AgentState
from codur.graph.node_types import ExecuteNodeResult, AgentOutcome
from codur.graph.state_operations import (
    get_llm_calls,
    get_messages,
    is_verbose,
    add_messages,
    normalize_messages
)
from codur.tools.meta_tools import VerificationResult
from codur.tools.schema_generator import get_function_schemas
from codur.tools.tool_annotations import ToolSideEffect
from codur.utils.llm_helpers import create_and_invoke_with_tool_support


"""
NOTE:
    - Verification agent uses tools to determine if implementation satisfies requirements
    - No hardcoded patterns (main.py, app.py, expected.txt)
    - Dynamically infers verification strategy from context
    - State operations should use codur.graph.state_operations
"""

console = Console()


# Build system prompt with available tools
def _get_system_prompt_with_tools():
    """Build system prompt with available tools listed."""
    from codur.tools.registry import list_tools_for_tasks
    from codur.constants import TaskType

    # Get verification-relevant tools
    # Match the include_unannotated setting with what verification_agent_node uses
    verification_task_types = [
        TaskType.CODE_VALIDATION,
        TaskType.RESULT_VERIFICATION,  # Includes build_verification_response
        TaskType.FILE_OPERATION,
        TaskType.EXPLANATION,
    ]
    # CRITICAL: Must match include_unannotated=True in verification_agent_node
    # Otherwise system prompt lists different tools than actual tool_schemas
    tools = list_tools_for_tasks(verification_task_types, include_unannotated=True)

    # Categorize tools by TaskType for readability
    discovery_tools = []
    execution_tools = []
    analysis_tools = []
    file_tools = []
    response_tools = []
    other_tools = []

    for tool in tools:
        name = tool['name']
        scenarios = tool.get('scenarios', [])

        if TaskType.RESULT_VERIFICATION in scenarios:
            # Response tools (build_verification_response)
            response_tools.append(name)
        elif TaskType.EXPLANATION in scenarios:
            # Discovery tools (entry points, project structure)
            discovery_tools.append(name)
        elif TaskType.CODE_VALIDATION in scenarios:
            # Execution and analysis tools
            if 'run_' in name or 'pytest' in name:
                execution_tools.append(name)
            else:
                analysis_tools.append(name)
        elif TaskType.FILE_OPERATION in scenarios:
            # File reading tools
            file_tools.append(name)
        else:
            other_tools.append(name)

    tools_section = f"""
## Available Tools

You have access to verification-relevant tools. Call them directly - they will be executed automatically.

**Discovery**: {', '.join(sorted(discovery_tools))}
**Execution**: {', '.join(sorted(execution_tools))}
**Analysis**: {', '.join(sorted(analysis_tools))}
**File Reading**: {', '.join(sorted(file_tools))}
**Response**: {', '.join(sorted(response_tools))}
{"**Other**: " + ', '.join(sorted(other_tools)) if other_tools else ""}

CRITICAL: Only use tools from this list. Do NOT invent or create new tools.
"""

    return f"""You are Codur Verification Agent, a specialized agent for verifying code implementations.

Your mission: Determine if the implemented code satisfies the user's original requirements by choosing and executing appropriate verification strategies.

## Key Principles

1. **Context-Driven Strategy**: Analyze the original request, project structure, and available artifacts to choose verification approach
2. **Multiple Verification Methods**: Consider tests, execution, static analysis, and output comparison
3. **Explicit Success Criteria**: Extract what "success" means from the original request
4. **Evidence-Based Decisions**: Run tools to gather evidence, then make a clear pass/fail determination

{tools_section}

## Your Task

Given:
- The original user request (what they asked for)
- The project structure (what files exist)
- Any execution history or errors

Do:
1. **Infer Success Criteria**: What does the user expect? (tests pass, specific output, no errors, etc.)
2. **Choose Strategy**: Which verification method(s) are most appropriate?
3. **Execute Verification**: Run the necessary tools to gather evidence
4. **Make Decision**: Based on evidence, explicitly state PASS or FAIL with reasoning

Dont:
1. You have not been given access to modify files. Focus solely on verification.
2. Do NOT invent or call tools not listed above.
3. Do not rerun tools that have already been executed - use existing results.

## Critical Rule: No Duplicate Tool Calls

## Final Step: Report Your Decision

After executing verification tools and analyzing results, you MUST call the
`build_verification_response` tool with your decision:

**For PASS**:
```
build_verification_response(
    passed=True,
    reasoning="Explain what evidence supports success - be specific about what you checked and what the results were"
)
```

Example:
```
build_verification_response(
    passed=True,
    reasoning="All tests in test_main.py passed. Executed main.py successfully with exit code 0."
)
```

**For FAIL**:
```
build_verification_response(
    passed=False,
    reasoning="Explain what evidence shows failure - be specific about the mismatch",
    expected="What was expected based on the original request",
    actual="What was observed from tool execution",
    suggestions="Specific, actionable advice on how to fix the issue"
)
```

Example:
```
build_verification_response(
    passed=False,
    reasoning="Test case_2 failed: expected output '5' but got '6'",
    expected="Output: 5",
    actual="Output: 6",
    suggestions="Check the factorial formula - should be n + (n-1)! not n * (n-1)!"
)
```

## Important Notes

- If tests exist (test_*.py files), prioritize running them over direct execution
- If expected output files exist, use them for comparison
- Always provide evidence (tool results) for your decision
- If a tool returns an error, treat that as verification failure evidence
- run_python_file returns std_out/std_err/return_code; non-zero return_code or std_err indicates failure in the python script. error field is reserved for tool execution failures. If the pythonscript returns a non-zero return_code, immediately call build_verification_response to record a failure.
- Focus on behavior verification: does the code do what the user asked for?
- If the original request is vague or does not specify success criteria, infer reasonable expectations based on common practices, sometimes this means code runs without errors
- Sometimes tools that were already run (eg. list_files, run_pytest, run_python_file) provide enough information to make a decision without further execution
- In the build_verification_response highlight what was wrong or missing and then put steps that help towards fixing the issue.
"""

# Initialize system prompt with tools
VERIFICATION_AGENT_SYSTEM_PROMPT = _get_system_prompt_with_tools()


@prepend_summary
def verification_agent_node(
    state: AgentState,
    config: CodurConfig,
    summary: str,
    recursion_depth: int = 0
) -> ExecuteNodeResult:
    """Run verification agent to determine if implementation satisfies requirements.

    This agent:
    1. Analyzes original request to infer success criteria
    2. Chooses appropriate verification strategy (tests, execution, static analysis)
    3. Executes verification tools (with recursive tool loop up to 3 iterations)
    4. Returns structured PASS/FAIL result with evidence

    Args:
        state: Current graph state with message history
        summary: Message summary for context based on prior interactions
        config: Runtime configuration
        recursion_depth: Current recursion depth for tool execution loop (default: 0)

    Returns:
        ExecuteNodeResult with verification outcome
    """
    agent_name = "agent:codur-verification"
    verbose = is_verbose(state)

    if verbose:
        depth_info = f" (recursion depth: {recursion_depth})" if recursion_depth > 0 else ""
        console.print(f"[bold cyan]Running verification agent{depth_info}...[/bold cyan]")

    # Get only safe, read-only tools that match the system prompt's task types
    # This ensures the LLM only sees tools we've told it about
    from codur.constants import TaskType
    verification_task_types = [
        TaskType.CODE_VALIDATION,
        TaskType.RESULT_VERIFICATION,  # Includes build_verification_response
        TaskType.FILE_OPERATION,
        TaskType.EXPLANATION,
    ]
    tool_schemas = get_function_schemas(
        task_types=verification_task_types,
        exclude_side_effects=[
            ToolSideEffect.FILE_MUTATION,  # No modifying files
            ToolSideEffect.STATE_CHANGE,   # No git/env changes
        ],
        include_unannotated=True,  # Include tools without side effect annotations
    )

    if recursion_depth == 0:
        new_messages = [
            SystemMessage(content=VERIFICATION_AGENT_SYSTEM_PROMPT + "\n\n" + summary),
        ]
    else:
        new_messages = []

    try:
        new_messages, _, execution_result = create_and_invoke_with_tool_support(
            config,
            new_messages,
            tool_schemas,
            profile_name=config.llm.default_profile,
            temperature=0.0,  # Low temperature for consistent verification
            invoked_by="verification.primary",
            state=state,
        )
    except Exception as exc:
        if verbose:
            console.log(f"[yellow]Verification invocation failed: {exc}[/yellow]")
        # Fallback: create ToolMessage with VerificationResult for consistent parsing
        error_result: VerificationResult = {
            "passed": False,
            "reasoning": f"Verification agent encountered an error: {str(exc)}",
            "expected": "Successful verification execution",
            "actual": "Exception during verification",
            "suggestions": "Check agent configuration and tool availability",
        }
        tool_result = {"tool": "build_verification_response", "output": error_result}
        tool_result_json = json.dumps(tool_result)
        tool_call_id = str(hash(tool_result_json))
        error_msg = ToolMessage(
            content=tool_result_json,
            tool_call_id=tool_call_id,
            name="build_verification_response",
        )

        return ExecuteNodeResult(
            agent_outcomes=[AgentOutcome(
                agent=agent_name,
                status="error",
                messages=new_messages + [error_msg],
                result=f"Verification error: {str(exc)}",
            )],
            messages=new_messages + [error_msg],
            llm_calls=get_llm_calls(state),
        )

    # Check if build_verification_response was called (final response)
    is_final_response = any(
        result.get("tool") == "build_verification_response"
        for result in (execution_result.results or [])
    )

    # If tools were called (but not final response), recurse to let agent process results
    if execution_result.results and recursion_depth <= 4 and not is_final_response:
        # Inject messages into state for next iteration
        add_messages(state, new_messages)

        # Recurse to let agent see tool results and make final decision
        nested_outcome = verification_agent_node(
            state=state,
            config=config,
            summary=summary,
            recursion_depth=recursion_depth + 1
        )
        nested_outcome["messages"] = new_messages + nested_outcome["messages"]
        return nested_outcome

    # Parse verification result from agent response
    verification_outcome = get_execution_result(execution_result)

    # Build agent outcome with verification result
    agent_outcome = AgentOutcome(
        agent=agent_name,
        messages=new_messages,
        status="success" if verification_outcome.get("passed") else "failed",
        result=verification_outcome.get("reasoning", "No reasoning provided"),
    )

    # Add suggestions to agent outcome if present
    if "suggestions" in verification_outcome:
        agent_outcome["next_step_suggestion"] = verification_outcome["suggestions"]

    return ExecuteNodeResult(
        agent_outcomes=[agent_outcome],
        messages=new_messages,
        llm_calls=get_llm_calls(state),
    )


def get_execution_result(execution_result) -> VerificationResult:
    """Parse verification outcome from tool call or agent messages.

    Looks for build_verification_response tool call in execution results.
    Falls back to JSON parsing if tool call not found.

    Returns:
        {
            "passed": bool,
            "reasoning": str,
            "expected": str | None,
            "actual": str | None,
            "suggestions": str | None,
            "raw_response": str,
        }
    """
    # Check if build_verification_response was called
    if execution_result and execution_result.results:
        for result in execution_result.results:
            if result.get("tool") == "build_verification_response":
                return result["output"]

    # Final fallback
    return {
        "passed": False,
        "reasoning": f"build_verification_response not yet called",
    }
