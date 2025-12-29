"""Dedicated verification node for the codur-verification agent."""
import re
from typing import Any

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
)
from codur.tools.schema_generator import get_function_schemas
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


# Whitelist of verification-relevant tools
ALLOWED_VERIFICATION_TOOLS = {
    'discover_entry_points',
    'get_primary_entry_point',
    'run_python_file',
    'run_pytest',
    'validate_python_syntax',
    'code_quality',
    'read_file',
    'read_files',
    'list_directory',
}


# Build system prompt with available tools
def _get_system_prompt_with_tools():
    """Build system prompt with available tools listed."""
    from codur.tools.registry import list_tools_for_tasks
    from codur.constants import TaskType

    # Get verification-relevant tools
    verification_task_types = [
        TaskType.CODE_VALIDATION,
        TaskType.FILE_OPERATION,
        TaskType.EXPLANATION,
    ]
    tools = list_tools_for_tasks(verification_task_types, include_unannotated=False)

    # Filter to whitelisted verification tools
    verification_tools = [t for t in tools if t['name'] in ALLOWED_VERIFICATION_TOOLS]

    # Categorize tools by TaskType for readability
    discovery_tools = []
    execution_tools = []
    analysis_tools = []
    file_tools = []

    for tool in verification_tools:
        name = tool['name']
        scenarios = tool.get('scenarios', [])

        if TaskType.EXPLANATION in scenarios:
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
   - Use discover_entry_points to find test files (test_*.py, *_test.py)
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

## Output Format

You MUST end your response with one of these formats:

**VERIFICATION: PASS**
Reasoning: [Explain what evidence supports success - be specific about what you checked and what the results were]

**VERIFICATION: FAIL**
Reasoning: [Explain what evidence shows failure - be specific about the mismatch or error]
Expected: [What was expected based on the original request]
Actual: [What was observed from tool execution]
Suggestions: [Specific, actionable advice on how to fix the issue]

## Important Notes

- If you cannot determine verification strategy from context, use hybrid approach (syntax check + execution attempt)
- If tests exist (test_*.py files), prioritize running them over direct execution
- If expected output files exist, use them for comparison
- Always provide evidence (tool results) for your decision
- Be explicit about what you checked and why
- If a tool returns an error, treat that as verification failure evidence
- Focus on behavior verification: does the code do what the user asked for?
"""

# Initialize system prompt with tools
VERIFICATION_AGENT_SYSTEM_PROMPT = _get_system_prompt_with_tools()
VERIFICATION_AGENT_SYSTEM_PROMPT_SUMMARY = (
    "Verification agent: determine if implementation satisfies requirements using context-driven strategies"
)


def _get_verification_tools() -> list[dict[str, Any]]:
    """Get tools relevant to verification tasks.

    Returns whitelisted verification-relevant tools only.
    """
    from codur.tools.registry import list_tools_for_tasks
    from codur.constants import TaskType

    # Verification-relevant task types
    verification_task_types = [
        TaskType.CODE_VALIDATION,
        TaskType.FILE_OPERATION,
        TaskType.EXPLANATION,  # For discovery tools
    ]

    all_tools = list_tools_for_tasks(verification_task_types, include_unannotated=False)

    # Filter to whitelisted verification tools
    return [t for t in all_tools if t['name'] in ALLOWED_VERIFICATION_TOOLS]


def verification_agent_node(
    state: AgentState,
    config: CodurConfig
) -> ExecuteNodeResult:
    """Run verification agent to determine if implementation satisfies requirements.

    This agent:
    1. Analyzes original request to infer success criteria
    2. Chooses appropriate verification strategy (tests, execution, static analysis)
    3. Executes verification tools
    4. Returns structured PASS/FAIL result with evidence

    Args:
        state: Current graph state with message history
        config: Runtime configuration

    Returns:
        ExecuteNodeResult with verification outcome
    """
    agent_name = "agent:codur-verification"
    verbose = is_verbose(state)

    if verbose:
        console.print("[bold cyan]Running verification agent...[/bold cyan]")

    # Build verification prompt with original request context
    prompt = _build_verification_prompt(get_messages(state))

    # Get verification-specific tools
    verification_tools = _get_verification_tools()
    tool_schemas = get_function_schemas()

    # Filter to only verification tools
    verification_tool_names = {t['name'] for t in verification_tools}
    filtered_schemas = [s for s in tool_schemas if s.get('name') in verification_tool_names]

    messages = [
        ShortenableSystemMessage(
            content=VERIFICATION_AGENT_SYSTEM_PROMPT,
            short_content=VERIFICATION_AGENT_SYSTEM_PROMPT_SUMMARY,
        ),
        HumanMessage(content=prompt),
    ]

    try:
        new_messages, execution_result = create_and_invoke_with_tool_support(
            config,
            messages,
            filtered_schemas,
            profile_name=config.llm.default_profile,
            temperature=0.0,  # Low temperature for consistent verification
            invoked_by="verification.primary",
            state=state,
        )
    except Exception as exc:
        if verbose:
            console.log(f"[yellow]Verification invocation failed: {exc}[/yellow]")
        # Fallback: assume failure with error message
        return {
            "agent_outcome": {
                "agent": agent_name,
                "result": f"Verification error: {str(exc)}",
                "status": "error",
                "verification_details": {
                    "passed": False,
                    "reasoning": f"Verification error: {str(exc)}",
                    "raw_response": "",
                }
            },
            "messages": [],
            "llm_calls": get_llm_calls(state),
        }

    # Parse verification result from agent response
    verification_outcome = _parse_verification_result(new_messages, execution_result)

    if verbose:
        status_str = "[green]PASS[/green]" if verification_outcome.get("passed") else "[red]FAIL[/red]"
        console.print(f"[bold cyan]Verification result: {status_str}[/bold cyan]")
        console.print(f"[dim]{verification_outcome.get('reasoning', 'No reasoning')[:200]}...[/dim]")

    return {
        "agent_outcome": {
            "agent": agent_name,
            "result": verification_outcome.get("reasoning", "No reasoning provided"),
            "status": "success" if verification_outcome.get("passed") else "failed",
            "verification_details": verification_outcome,
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
    """
    normalized = normalize_messages(messages)

    original_request = None
    implementation_summary = []
    prior_verifications = []

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

    if prior_verifications:
        prompt_parts.extend([
            "## Prior Verification Attempts",
            "Previous verifications have been attempted. Review them to understand what was already checked:",
            *prior_verifications[-2:],  # Only include last 2 to avoid context bloat
            "",
        ])

    prompt_parts.extend([
        "## Instructions",
        "1. Use discovery tools to understand project structure (entry points, tests)",
        "2. Infer what 'success' means from the original request",
        "3. Choose and execute appropriate verification strategy",
        "4. Make explicit PASS/FAIL decision with evidence",
        "",
        "Start by discovering the project structure, then choose your verification approach.",
    ])

    return "\n".join(prompt_parts)


def _parse_verification_result(messages, execution_result) -> dict:
    """Parse verification outcome from agent messages.

    Looks for:
    - **VERIFICATION: PASS** or **VERIFICATION: FAIL** markers
    - Reasoning, Expected, Actual, Suggestions fields

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
    # Combine all AI messages into one response
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    if not ai_messages:
        return {
            "passed": False,
            "reasoning": "No verification response generated",
            "raw_response": "",
        }

    full_response = "\n".join(m.content for m in ai_messages)

    # Parse structured output
    passed = "VERIFICATION: PASS" in full_response
    failed = "VERIFICATION: FAIL" in full_response

    if not passed and not failed:
        # Agent didn't follow format - default to fail
        return {
            "passed": False,
            "reasoning": f"Verification agent did not provide explicit PASS/FAIL decision. Response: {full_response[:200]}...",
            "raw_response": full_response,
        }

    # Extract fields using simple parsing
    reasoning = _extract_field(full_response, "Reasoning:")
    expected = _extract_field(full_response, "Expected:")
    actual = _extract_field(full_response, "Actual:")
    suggestions = _extract_field(full_response, "Suggestions:")

    return {
        "passed": passed,
        "reasoning": reasoning or "No reasoning provided",
        "expected": expected,
        "actual": actual,
        "suggestions": suggestions,
        "raw_response": full_response,
    }


def _extract_field(text: str, field_name: str) -> str | None:
    """Extract content after field marker until next field or end."""
    pattern = rf"{re.escape(field_name)}\s*(.+?)(?=\n(?:Reasoning:|Expected:|Actual:|Suggestions:|\*\*VERIFICATION:)|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None
