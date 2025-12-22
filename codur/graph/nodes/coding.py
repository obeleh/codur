"""Dedicated coding node for the codur-coding agent."""

import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import ExecuteNodeResult
from codur.graph.nodes.utils import _normalize_messages
from codur.llm import create_llm_profile, create_llm
from codur.tools import write_file

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

Return the complete solution as a single fenced code block with the language tag:
```python
<full file content>
```

No additional commentary outside the code block. Always include the backticks and the `python` language tag.

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
    response = llm.invoke(messages)
    result = response.content

    if not _extract_code_block(result):
        # One strict retry to force code-only output with a python code block.
        strict_messages = [
            SystemMessage(content=CODING_AGENT_SYSTEM_PROMPT),
            SystemMessage(content="Output only a single ```python fenced code block with the full file. No other text."),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(strict_messages)
        result = response.content

    _apply_coding_result(result, state, config)

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
        }
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


def _apply_coding_result(result: str, state: AgentState, config: CodurConfig) -> None:
    """Apply a code block result to main.py when present."""
    code_block = _extract_code_block(result)
    if not code_block:
        return

    target = Path.cwd() / "main.py"
    if not target.exists():
        return

    write_file(
        path=str(target),
        content=code_block,
        root=Path.cwd(),
        allow_outside_root=config.runtime.allow_outside_workspace,
        state=state,
    )


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()
