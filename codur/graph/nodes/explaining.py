"Dedicated explaining node for the codur-explaining agent."
import json
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.nodes.types import ExecuteNodeResult
from codur.graph.nodes.utils import normalize_messages, resolve_llm_for_model
from codur.llm import create_llm_profile
from codur.utils.llm_calls import invoke_llm

console = Console()


# Built-in system prompt for the explaining agent
EXPLAINING_AGENT_SYSTEM_PROMPT = """You are Codur Explaining Agent, a specialized technical communicator.

Your mission: Explain code, architecture, and technical concepts clearly and accurately.

## Key Principles

1. **Context-Aware**: Use the provided file contents and context to ground your explanation.
2. **Structure**: Use Markdown headers, bullet points, and code blocks for readability.
3. **Level of Detail**: Adapt to the query. If asked "what does this do?", provide a summary. If asked for details, go deep.
4. **Accuracy**: Do not hallucinate code that isn't in the context. If you don't know, state what is missing.

## Output Format

Return the explanation in clear Markdown format.
"""


def explaining_node(state: AgentState, config: CodurConfig) -> ExecuteNodeResult:
    """Run the codur-explaining agent.

    Args:
        state: Current graph state with messages, iterations, etc.
        config: Runtime configuration

    Returns:
        ExecuteNodeResult with agent_outcome
    """
    agent_name = "agent:codur-explaining"
    iterations = state.get("iterations", 0)
    verbose = state.get("verbose", False)

    if verbose:
        console.print(f"[bold blue]Running codur-explaining node (iteration {iterations})...[/bold blue]")

    # Resolve LLM (uses default LLM)
    # Use standard temperature for explanation (balanced creativity/accuracy)
    llm = resolve_llm_for_model(
        config,
        None,
        temperature=0.7,
        json_mode=False
    )

    # Build context-aware prompt
    prompt = _build_explaining_prompt(state.get("messages", []))

    # Use built-in system prompt
    messages = [
        SystemMessage(content=EXPLAINING_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Invoke LLM
    if verbose:
        console.log("[bold cyan]Invoking codur-explaining LLM...[/bold cyan]")
    
    response = invoke_llm(
        llm,
        messages,
        invoked_by="explaining.primary",
        state=state,
        config=config,
    )

    result = response.content

    if verbose:
        console.print(f"[dim]codur-explaining response length: {len(result)} chars[/dim]")
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


def _build_explaining_prompt(raw_messages) -> str:
    """Build context-aware prompt from graph state messages."""
    messages = normalize_messages(raw_messages)

    challenge = None
    context_parts = []

    for message in messages:
        if isinstance(message, HumanMessage) and challenge is None:
            challenge = message.content
        elif isinstance(message, SystemMessage):
            content = message.content
            # Filter out verification errors if they exist (though unlikely for explanation)
            if "Verification failed" not in content and "=== Expected Output ===" not in content:
                context_parts.append(content)
        elif isinstance(message, HumanMessage):
            context_parts.append(message.content)
        elif isinstance(message, AIMessage):
            # Include previous AI responses as context if relevant
            context_parts.append(f"Previous response: {message.content}")

    if challenge is None:
        challenge = messages[-1].content if messages else "No task provided."

    if context_parts:
        context_text = "\n\n---\n\n".join(context_parts)
        return f"EXPLANATION REQUEST:\n{challenge}\n\nADDITIONAL CONTEXT:\n{context_text}"
    else:
        return challenge
