"Dedicated explaining node for the codur-explaining agent."

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from rich.console import Console

from codur.config import CodurConfig
from codur.graph.state import AgentState
from codur.graph.node_types import ExecuteNodeResult
from codur.graph.utils import normalize_messages
from codur.graph.state_operations import get_iterations, get_llm_calls, get_messages, is_verbose
from codur.utils.llm_helpers import create_and_invoke

console = Console()


# Built-in system prompt for the explaining agent
EXPLAINING_AGENT_SYSTEM_PROMPT = """You are Codur Code Explainer Agent, a specialized technical communicator for code analysis and documentation.

Your mission: Explain code, architecture, and technical concepts with precision and clarity.

## Key Principles

1. **Context-Aware**: Ground explanations in the provided file contents, AST information, and dependency graphs.
2. **Structure**: Use clear Markdown formatting with headers, bullet points, code blocks, and diagrams when helpful.
3. **Level of Detail**: Adapt to the query complexity:
   - Simple "what does this do?" → High-level summary with key points
   - "Explain how X works" → Detailed flow with code examples
   - "Describe the architecture" → System overview with component relationships
4. **Accuracy**: Only reference code that exists in the provided context. If information is missing, explicitly state what's needed.
5. **Code Flow**: When explaining logic, trace execution paths and data transformations clearly.
6. **Dependencies**: Highlight imports, function calls, and module relationships when relevant.

## Analysis Approach

When AST or dependency information is available:
- Use function/class definitions to explain structure
- Reference dependency graphs to show relationships
- Identify key patterns and design decisions
- Note potential issues or improvements (if asked)

## Output Format

Provide explanations in well-structured Markdown:
- Start with a brief summary (1-2 sentences)
- Use sections for different aspects (Purpose, Key Components, Flow, etc.)
- Include code snippets with syntax highlighting when referencing specific implementations
- End with context or related information if helpful

## Examples of Good Explanations

**Good**: "The `title_case()` function converts strings to title case by capitalizing the first letter of each word. It handles edge cases including hyphenated words (e.g., 'state-of-the-art' → 'State-Of-The-Art') and preserves all-caps acronyms like NASA."

**Bad**: "This function does some string manipulation." (too vague)

**Good**: "The authentication flow consists of three phases: (1) credential validation in `auth.py:validate_user()`, (2) JWT token generation in `tokens.py:create_token()`, and (3) session storage in Redis via the `SessionManager` class."

**Bad**: "It handles authentication." (lacks detail and structure)
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
    iterations = get_iterations(state)
    verbose = is_verbose(state)

    if verbose:
        console.print(f"[bold blue]Running codur-explaining node (iteration {iterations})...[/bold blue]")

    # Build context-aware prompt
    prompt = _build_explaining_prompt(get_messages(state))

    # Use built-in system prompt
    messages = [
        SystemMessage(content=EXPLAINING_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Invoke LLM
    if verbose:
        console.log("[bold cyan]Invoking codur-explaining LLM...[/bold cyan]")
    
    response = create_and_invoke(
        config,
        messages,
        temperature=0.5,
        invoked_by="explaining.primary",
        state=state,
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
        "llm_calls": get_llm_calls(state),
    }


def _build_explaining_prompt(raw_messages) -> str:
    """Build context-aware prompt from graph state messages.

    Extracts and organizes:
    - User's explanation request
    - File contents and code snippets
    - AST information (function/class definitions, dependencies)
    - Tool results (file trees, grep results, etc.)
    - Previous conversation context
    """
    messages = normalize_messages(raw_messages)

    challenge = None
    file_contents = []
    ast_info = []
    tool_results = []
    conversation_context = []

    for message in messages:
        if isinstance(message, HumanMessage):
            if challenge is None:
                challenge = message.content
            else:
                # Additional user clarifications
                conversation_context.append(f"User clarification: {message.content}")
        elif isinstance(message, SystemMessage):
            content = message.content
            # Filter out verification errors (not relevant for explanations)
            if "Verification failed" in content or "=== Expected Output ===" in content:
                continue

            # Categorize system messages by type
            if "Tool results:" in content:
                # Extract tool results
                tool_results.append(content)
            elif any(marker in content for marker in ["def ", "class ", "import ", "from "]):
                # Looks like file content or code
                file_contents.append(content)
            elif any(marker in content for marker in ["AST", "dependencies", "function:", "class:"]):
                # AST or dependency information
                ast_info.append(content)
            else:
                # General context
                conversation_context.append(content)
        elif isinstance(message, AIMessage):
            # Include previous AI responses for multi-turn conversations
            content_str = message.content if isinstance(message.content, str) else str(message.content)
            conversation_context.append(f"Previous explanation: {content_str}")

    if challenge is None:
        # Fallback: use last message content
        last_msg = messages[-1] if messages else None
        if last_msg:
            challenge = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        else:
            challenge = "No explanation request provided."

    # Build structured prompt
    prompt_parts = [f"EXPLANATION REQUEST:\n{challenge}"]

    if file_contents:
        prompt_parts.append(f"\nFILE CONTENTS:\n{chr(10).join(file_contents)}")

    if ast_info:
        prompt_parts.append(f"\nCODE STRUCTURE (AST/Dependencies):\n{chr(10).join(ast_info)}")

    if tool_results:
        prompt_parts.append(f"\nTOOL RESULTS:\n{chr(10).join(tool_results)}")

    if conversation_context:
        prompt_parts.append(f"\nADDITIONAL CONTEXT:\n{chr(10).join(conversation_context)}")

    return "\n\n---\n\n".join(prompt_parts)
