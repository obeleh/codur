"""Dedicated coding node for the codur-coding agent."""
import os
import re
import json
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import ExecuteNodeResult, ReplacementDirective
from codur.graph.nodes.utils import _normalize_messages
from codur.llm import create_llm_profile, create_llm
from codur.tools import write_file, read_file, replace_lines
from codur.tools.validation import validate_python_syntax
from codur.tools.ast_utils import find_function_lines, find_class_lines, find_method_lines
from codur.utils.llm_calls import invoke_llm

console = Console()


# Built-in system prompt for the coding agent
CODING_AGENT_SYSTEM_PROMPT = """You are Codur Coding Agent, a specialized coding solver.

Your mission: Solve coding challenges end-to-end with correct, efficient, and robust implementations.

## Key Principles

1. **Understand Requirements**: Carefully read the challenge and any provided context
2. **Edge Cases First**: Consider boundary conditions, empty inputs, large inputs, special characters
3. **Correctness Over Cleverness**: Prioritize working code over premature optimization
4. **Learn from Errors**: If this is a retry, analyze the verification failure carefully

## When This is a Retry (you'll see "Verification failed" or "PREVIOUS ATTEMPT FAILED"):

- **Compare expected vs actual output**: Identify exactly what differs
- **Check for common bugs**:
  - Off-by-one errors (range boundaries, string indices)
  - Missing edge cases (empty input, single element, null/None)
  - String formatting issues (trailing spaces, newlines, capitalization)
  - Type mismatches (int vs float, string vs number)
  - Logic inversions (< instead of <=, and vs or)
- **Review the implementation**: If provided, identify the specific line(s) causing the issue
- **Fix precisely**: Don't rewrite everything, fix the identified bug

## Output Format

You can respond in two ways:

### 1. FULL FILE REPLACEMENT (default, for complete implementations)
Return the entire file content in a ```python fenced code block:

```python
<full file content>
```

### 2. TARGETED REPLACEMENT (for single function/class updates)
When updating a specific function or class, use this format:

Replace function `function_name` with:
```python
def function_name(args):
    # implementation
```

Or for classes:

Replace class `ClassName` with:
```python
class ClassName:
    # implementation
```

**Important**: For targeted replacements:
- The function/class must already exist in the file
- Provide complete, syntactically valid code
- Include proper indentation

## Clarifications

Only ask for clarification if the task is genuinely ambiguous. If requirements are clear, proceed with implementation.
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
    llm = _resolve_llm_for_model(config, None)

    # Build context-aware prompt
    prompt = _build_coding_prompt(state.get("messages", []), iterations)

    # Use built-in system prompt
    messages = [
        SystemMessage(content=CODING_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Invoke LLM
    if verbose:
        console.log("[bold cyan]Invoking codur-coding LLM...[/bold cyan]")
    response = invoke_llm(
        llm,
        messages,
        invoked_by="coding.primary",
        state=state,
        config=config,
    )
    result = response.content

    if not _extract_code_block(result):
        # One strict retry to force code-only output with a python code block.
        strict_messages = [
            SystemMessage(content=CODING_AGENT_SYSTEM_PROMPT),
            SystemMessage(content="Output only a single ```python fenced code block with the full file. No other text."),
            HumanMessage(content=prompt),
        ]
        response = invoke_llm(
            llm,
            strict_messages,
            invoked_by="coding.strict",
            state=state,
            config=config,
        )
        result = response.content

    # Apply result and check for validation errors
    error = _apply_coding_result(result, state, config)
    if error:
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
            content=f"Previous attempt failed:\n{error}\n\nPlease return valid, complete Python code in a ```python fenced code block or use a targeted replacement format."
        )
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


def _resolve_llm_for_model(config: CodurConfig, model: str | None):
    """Resolve LLM instance from model identifier.

    Args:
        config: Runtime configuration
        model: Model identifier (e.g., "qwen/qwen3-32b")

    Returns:
        LLM instance
    """
    matching_profile = None
    if model:
        for profile_name, profile in config.llm.profiles.items():
            if profile.model == model:
                matching_profile = profile_name
                break
    return create_llm_profile(config, matching_profile) if matching_profile else create_llm(config)


def _build_coding_prompt(raw_messages, iterations: int = 0) -> str:
    """Build context-aware prompt from graph state messages.

    Handles both fresh attempts and retry scenarios with verification errors.

    Args:
        raw_messages: Message history from state
        iterations: Current retry iteration count

    Returns:
        Formatted prompt with challenge and relevant context
    """
    messages = _normalize_messages(raw_messages)

    # Extract components
    challenge = None
    verification_errors = []
    previous_attempts = []
    context_parts = []

    for message in messages:
        if isinstance(message, HumanMessage) and challenge is None:
            # First HumanMessage is the original task/challenge
            challenge = message.content

        elif isinstance(message, SystemMessage):
            content = message.content

            # Detect verification error
            if "Verification failed" in content or "=== Expected Output ===" in content:
                verification_errors.append(content)
            else:
                context_parts.append(content)

        elif isinstance(message, AIMessage):
            # Track previous attempts for context
            previous_attempts.append(message.content)

        elif isinstance(message, HumanMessage):
            # Additional human context
            context_parts.append(message.content)

    # Fallback if no challenge found
    if challenge is None:
        challenge = messages[-1].content if messages else "No task provided."

    # Build prompt based on context
    if verification_errors:
        # RETRY SCENARIO - use most recent error
        return _build_retry_prompt(challenge, verification_errors[-1], iterations)
    elif context_parts:
        # FIRST ATTEMPT WITH CONTEXT
        context_text = "\n\n---\n\n".join(context_parts)
        return f"CODING CHALLENGE:\n{challenge}\n\nADDITIONAL CONTEXT:\n{context_text}"
    else:
        # SIMPLE FIRST ATTEMPT
        return challenge


def _build_retry_prompt(challenge: str, verification_error: str, iterations: int) -> str:
    """Build focused prompt for retry attempts with error context.

    Provides iteration-specific guidance to help agent learn from mistakes.

    Args:
        challenge: Original coding challenge
        verification_error: Structured error message from review node
        iterations: Current retry count

    Returns:
        Prompt emphasizing error analysis and targeted fix
    """
    prompt_parts = [
        f"CODING CHALLENGE (Retry Attempt {iterations}):",
        challenge,
        "",
        "PREVIOUS ATTEMPT FAILED VERIFICATION:",
        verification_error,
        "",
        "INSTRUCTIONS FOR THIS RETRY:",
    ]

    # Provide iteration-specific guidance
    if iterations <= 2:
        prompt_parts.append(
            "- Carefully compare the Expected vs Actual output above\n"
            "- Identify the specific difference (missing line, wrong format, logic error)\n"
            "- Fix the precise issue without rewriting unrelated code"
        )
    elif iterations <= 5:
        prompt_parts.append(
            f"- This is retry attempt {iterations}. Previous attempts failed.\n"
            "- Check for common bugs: off-by-one errors, edge cases, string formatting\n"
            "- Review the Current Implementation section if provided\n"
            "- Focus on what's different between expected and actual"
        )
    else:
        prompt_parts.append(
            "- Multiple attempts have failed. Be very systematic:\n"
            "- Step 1: Identify EXACTLY what's wrong (line-by-line comparison)\n"
            "- Step 2: Check edge cases (empty input, single element, boundaries)\n"
            "- Step 3: Verify string formatting (spaces, newlines, capitalization)\n"
            "- Step 4: Implement the fix and DOUBLE CHECK your logic"
        )

    return "\n".join(prompt_parts)


def _parse_replacement_directive(text: str) -> Optional[ReplacementDirective]:
    """
    Parse replacement directive from coding agent response.

    Supports formats:
    1. "Replace function `name` with:" followed by code block
    2. "Replace class `Name` with:" followed by code block
    3. "Update method `ClassName.method_name`:" followed by code block
    4. JSON format: {"operation": "replace_function", "name": "...", "code": "..."}

    Returns:
        ReplacementDirective if found, None otherwise
    """
    # Try JSON format first
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if 'operation' in data and 'name' in data and 'code' in data:
                return ReplacementDirective(
                    operation=data['operation'],  # type: ignore
                    target_name=data['name'],
                    class_name=data.get('class_name'),
                    code=data['code']
                )
        except (json.JSONDecodeError, KeyError):
            pass

    # Try natural language patterns
    # Pattern: "Replace function `name` with:" or "Replace function name with:"
    func_match = re.search(
        r'Replace function\s+[`\']?(\w+)[`\']?\s+with:',
        text,
        re.IGNORECASE
    )
    if func_match:
        code = _extract_code_block(text)
        if code:
            return ReplacementDirective(
                operation="replace_function",  # type: ignore
                target_name=func_match.group(1),
                code=code
            )

    # Pattern: "Replace class `Name` with:"
    class_match = re.search(
        r'Replace class\s+[`\']?(\w+)[`\']?\s+with:',
        text,
        re.IGNORECASE
    )
    if class_match:
        code = _extract_code_block(text)
        if code:
            return ReplacementDirective(
                operation="replace_class",  # type: ignore
                target_name=class_match.group(1),
                code=code
            )

    # Pattern: "Update method `ClassName.method_name`:" or "Update method ClassName.method_name:"
    method_match = re.search(
        r'Update method\s+[`]?(\w+)\.(\w+)[`]?\s+with:',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if method_match:
        code = _extract_code_block(text)
        if code:
            return ReplacementDirective(
                operation="replace_method",  # type: ignore
                target_name=method_match.group(2),
                class_name=method_match.group(1),
                code=code
            )

    return None


def _apply_coding_result(result: str, state: AgentState, config: CodurConfig) -> Optional[str]:
    """
    Apply coding result to file using either:
    1. Targeted replacement (replace specific function/class)
    2. Full file write (default)

    Args:
        result: The LLM response containing code
        state: Current graph state
        config: Runtime configuration

    Returns:
        Error message if validation/application fails, None on success
    """
    # Check if target file exists
    target_file = Path.cwd() / "main.py"
    if not target_file.exists():
        return None  # File doesn't exist, skip application

    # Parse directive
    directive = _parse_replacement_directive(result)

    if directive:
        # TARGETED REPLACEMENT PATH
        code = directive["code"]

        # Validate syntax
        is_valid, error_msg = validate_python_syntax(code)
        if not is_valid:
            return f"Invalid Python syntax in replacement code:\n{error_msg}"

        try:
            # Read current file
            current_content = read_file("main.py")

            # Find line range based on operation type
            if directive["operation"] == "replace_function":
                line_range = find_function_lines(current_content, directive["target_name"])
                target_desc = f"function {directive['target_name']}"
            elif directive["operation"] == "replace_class":
                line_range = find_class_lines(current_content, directive["target_name"])
                target_desc = f"class {directive['target_name']}"
            elif directive["operation"] == "replace_method":
                if not directive.get("class_name"):
                    return "Method replacement requires class_name"
                line_range = find_method_lines(
                    current_content,
                    directive["class_name"],
                    directive["target_name"]
                )
                target_desc = f"method {directive['class_name']}.{directive['target_name']}"
            else:
                return f"Unknown operation: {directive['operation']}"

            if not line_range:
                return f"Could not find {target_desc} in main.py"

            start_line, end_line = line_range

            # Apply replacement
            replace_lines(
                path="main.py",
                start_line=start_line,
                end_line=end_line,
                content=code,
                root=Path.cwd(),
                allow_outside_root=config.runtime.allow_outside_workspace,
            )


            if os.environ.get("EARLY_FAILURE_HELPERS_FOR_TESTS") == "1":
                # read file and make sure if if __name__ == "__main__": still is in the file
                updated_content = read_file("main.py")
                if '__name__ == "__main__":' not in updated_content:
                    return "Replacement removed the main entry point."

            return None  # Success
        except Exception as e:
            return f"Failed to apply replacement: {str(e)}"

    else:
        # FULL FILE WRITE PATH (existing behavior)
        code_block = _extract_code_block(result)
        if not code_block:
            return "No valid ```python code block found in response."

        # Validate syntax
        is_valid, error_msg = validate_python_syntax(code_block)
        if not is_valid:
            return f"Invalid Python syntax:\n{error_msg}"

        # Write full file
        try:
            write_file(
                path="main.py",
                content=code_block,
                root=Path.cwd(),
                allow_outside_root=config.runtime.allow_outside_workspace,
                state=state,
            )
            return None  # Success
        except Exception as e:
            return f"Failed to write file: {str(e)}"


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()
