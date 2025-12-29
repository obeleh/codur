"""Dedicated verification node for the codur-verification agent."""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.node_types import ExecuteNodeResult
from codur.graph.utils import normalize_messages
from codur.graph.state_operations import (
    get_llm_calls,
    get_messages,
    is_verbose,
    add_messages,
)
from codur.graph.planning.json_parser import JSONResponseParser
from codur.tools.schema_generator import get_function_schemas
from codur.tools.tool_annotations import ToolSideEffect
from codur.utils.llm_helpers import (
    ShortenableSystemMessage,
    create_and_invoke_with_tool_support,
)


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
            # Fallback categorization for edge cases
            if 'entry_point' in name or 'discover' in name:
                discovery_tools.append(name)
            elif 'run_' in name:
                execution_tools.append(name)
            elif 'read' in name or 'list' in name:
                file_tools.append(name)

    tools_section = f"""
## Available Tools

You have access to verification-relevant tools. Call them directly - they will be executed automatically.

**Discovery**: {', '.join(sorted(discovery_tools))}
**Execution**: {', '.join(sorted(execution_tools))}
**Analysis**: {', '.join(sorted(analysis_tools))}
**File Reading**: {', '.join(sorted(file_tools))}
**Response**: {', '.join(sorted(response_tools))}

CRITICAL: Only use tools from this list. Do NOT invent or create new tools.
"""

    return f"""You are Codur Verification Agent, a specialized agent for verifying code implementations.

Your mission: Determine if the implemented code satisfies the user's original requirements by choosing and executing appropriate verification strategies.

## Key Principles

1. **Context-Driven Strategy**: Analyze the original request, project structure, and available artifacts to choose verification approach
2. **Multiple Verification Methods**: Consider tests, execution, static analysis, and output comparison
3. **Explicit Success Criteria**: Extract what "success" means from the original request
4. **Evidence-Based Decisions**: Run tools to gather evidence, then make a clear pass/fail determination

## Available Verification Strategies

1. **Test-Based Verification** (Preferred when tests exist)
   - Use list_files find test files (test_*.py, *_test.py)
   - Use run_pytest to execute tests with appropriate filters
   - Best for: Projects with test suites, TDD challenges, library implementations

2. **Execution-Based Verification** (When output is specified or describable)
   - Use discover_entry_points or get_primary_entry_point to find executable files
   - Use run_python_file to execute and capture output
   - Use read_file to check expected output files if they exist
   - Best for: CLI tools, scripts with specified output, algorithm challenges

3. **Static Analysis Verification** (For code quality tasks)
   - Use validate_python_syntax for syntax correctness
   - Use code_quality for linting/style requirements
   - Best for: Refactoring tasks, style compliance, structural requirements

4. **Hybrid Verification** (Combine multiple strategies)
   - Run tests AND check output
   - Validate syntax AND run execution
   - Best for: Complex requirements with multiple success criteria
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

## Critical Rule: No Duplicate Tool Calls

⚠️ IMPORTANT: If you see a tool in the "Previous Tool Execution Results" section above,
DO NOT CALL IT AGAIN. You already have its results. Use the existing results to make
your final decision instead of repeating the tool call.

Calling the same tool twice wastes time and tokens. Analyze what you already know.

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
- Be explicit about what you checked and why
- If a tool returns an error, treat that as verification failure evidence
- Focus on behavior verification: does the code do what the user asked for?
- If the original request is vague or does not specify success criteria, infer reasonable expectations based on common practices, sometimes this means code runs without errors
- Sometimes tools that were already run (eg. list_files, run_pytest, run_python_file) provide enough information to make a decision without further execution
"""

# Initialize system prompt with tools
VERIFICATION_AGENT_SYSTEM_PROMPT = _get_system_prompt_with_tools()
VERIFICATION_AGENT_SYSTEM_PROMPT_SUMMARY = (
    "Verification agent: determine if implementation satisfies requirements using context-driven strategies"
)


def verification_agent_node(
    state: AgentState,
    config: CodurConfig,
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

    # Build verification prompt with original request context
    prompt = _build_verification_prompt(get_messages(state))

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

    new_messages = [
        ShortenableSystemMessage(
            content=VERIFICATION_AGENT_SYSTEM_PROMPT,
            short_content=VERIFICATION_AGENT_SYSTEM_PROMPT_SUMMARY,
            long_form_visible_for_agent_name="verification",
        ),
        HumanMessage(content=prompt),
    ]

    try:
        new_messages, execution_result = create_and_invoke_with_tool_support(
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
        # Fallback: create error message with VERIFICATION: FAIL format
        error_msg = AIMessage(content=f"""**VERIFICATION: FAIL**
Reasoning: Verification agent encountered an error: {str(exc)}
Expected: Successful verification execution
Actual: Exception during verification
Suggestions: Check agent configuration and tool availability""")

        return {
            "agent_outcome": {
                "agent": agent_name,
                "result": f"Verification error: {str(exc)}",
                "status": "error",
            },
            "messages": new_messages + [error_msg],
            "llm_calls": get_llm_calls(state),
        }

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
        nested_outcome = verification_agent_node(state, config, recursion_depth + 1)
        nested_outcome["messages"] = new_messages + nested_outcome["messages"]
        return nested_outcome

    # Parse verification result from agent response
    verification_outcome = _parse_verification_result(new_messages, execution_result)
    return {
        "agent_outcome": {
            "agent": agent_name,
            "result": verification_outcome.get("reasoning", "No reasoning provided"),
            "status": "success" if verification_outcome.get("passed") else "failed",
        },
        "messages": new_messages,
        "llm_calls": get_llm_calls(state),
    }


def _build_verification_prompt(messages) -> str:
    """Build verification prompt from message history.

    Extracts:
    - Original user request (first HumanMessage)
    - Implementation history (AIMessages, SystemMessages with code changes)
    - Any prior verification attempts
    - Tool execution results (ToolMessages from recursive calls)
    """
    from langchain_core.messages import ToolMessage

    normalized = normalize_messages(messages)

    original_request = None
    implementation_summary = []
    prior_verifications = []
    tool_results = []

    for msg in normalized:
        if isinstance(msg, HumanMessage) and original_request is None:
            original_request = msg.content
        elif isinstance(msg, AIMessage):
            # Check if this message contains implementation work
            content_lower = msg.content.lower()
            if any(indicator in content_lower for indicator in ['write_file', 'replace_function', 'tool calls', 'implementation']):
                implementation_summary.append("Code implementation work completed")
        elif isinstance(msg, SystemMessage):
            if "Verification failed" in msg.content or "VERIFICATION:" in msg.content:
                prior_verifications.append(msg.content)
        elif isinstance(msg, ToolMessage):
            # Include tool execution results so agent knows what was already executed
            tool_name = msg.name or "unknown_tool"
            tool_results.append(f"**{tool_name}**: {msg.content[:200]}")

    if original_request is None:
        original_request = normalized[-1].content if normalized else "No original request found in message history"

    prompt_parts = [
        "# Verification Task",
        "",
        "## Original User Request",
        original_request,
        "",
        "## Your Task",
        "Verify that the current implementation satisfies the original user request.",
        "Use the appropriate verification strategy based on project context.",
        "",
    ]

    if tool_results:
        prompt_parts.extend([
            "## Previous Tool Execution Results",
            "These tools were already executed in previous attempts. You already have this information:",
            *tool_results,
            "",
            "IMPORTANT: Do NOT call the same tools again. Use these results to analyze and make your final decision.",
            "",
        ])

    if prior_verifications:
        prompt_parts.extend([
            "## Prior Verification Attempts",
            "Previous verifications have been attempted. Review them to understand what was already checked:",
            *prior_verifications[-2:],  # Only include last 2 to avoid context bloat
            "",
        ])

    prompt_parts.extend([
        "## Instructions",
        "1. Use discovery tools to understand project structure (list files, entry points)",
        "2. Infer what 'success' means from the original request",
        "3. Choose and execute appropriate verification strategy",
        "4. Make explicit PASS/FAIL decision with evidence",
        "",
        "Start by discovering the project structure, then choose your verification approach.",
    ])

    return "\n".join(prompt_parts)


def _parse_verification_result(messages, execution_result) -> dict:
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
                args = result.get("args", {})
                # Get boolean value directly from args
                passed = args.get("passed", False)

                return {
                    "passed": bool(passed),
                    "reasoning": args.get("reasoning", "No reasoning provided"),
                    "expected": args.get("expected"),
                    "actual": args.get("actual"),
                    "suggestions": args.get("suggestions"),
                    "raw_response": str(args),
                }


    # Final fallback
    return {
        "passed": False,
        "reasoning": f"build_verification_response not yet called",
    }
