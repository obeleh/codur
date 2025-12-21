"""Planning logic for the graph nodes."""

from typing import Dict, Any, Optional
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.tools.registry import list_tool_directory
from codur.graph.nodes.types import PlanNodeResult, PlanningDecision
from codur.graph.nodes.utils import _normalize_messages
from codur.graph.nodes.non_llm_tools import run_non_llm_tools

console = Console()


def clean_json_response(response: str) -> str:
    """Clean JSON response by removing artifacts before/after braces.

    LLMs sometimes add text before/after JSON. This function extracts
    only the JSON object from first '{' to last '}'.

    Args:
        response: Raw LLM response that should contain JSON

    Returns:
        str: Cleaned JSON string starting with { and ending with }

    Examples:
        >>> clean_json_response('Here is the result: {"key": "value"}')
        '{"key": "value"}'
        >>> clean_json_response('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
    """
    # Find first { and last }
    first_brace = response.find('{')
    last_brace = response.rfind('}')

    # If no braces found, return as-is (will fail parsing later)
    if first_brace == -1 or last_brace == -1:
        return response.strip()

    # Extract only the JSON portion
    return response[first_brace:last_brace + 1]


def _build_planning_system_prompt(config: CodurConfig) -> str:
    """Build the planning system prompt with config defaults and available tools.

    Args:
        config: Codur configuration

    Returns:
        str: Complete planning system prompt
    """
    # Get default agent from config
    default_agent = config.agents.preferences.default_agent
    if not default_agent:
        raise ValueError("agents.preferences.default_agent must be configured")

    # Get available tools
    tools = list_tool_directory()
    tool_names = [t['name'] for t in tools if isinstance(t, dict) and 'name' in t]

    # Build tool list for prompt
    file_tools = [t for t in tool_names if any(x in t for x in ['file', 'move', 'copy', 'read', 'write', 'delete'])]
    other_tools = [t for t in tool_names if t not in file_tools]

    tools_section = "**Available Tools:**\n"
    tools_section += "\nFile Operations:\n"
    tools_section += ", ".join(file_tools[:15])  # Limit to avoid token overflow
    if other_tools:
        tools_section += "\n\nOther Tools:\n"
        tools_section += ", ".join(other_tools[:15])

    return f"""You are Codur, an autonomous coding agent orchestrator.

**CRITICAL RULES - READ FIRST:**
1. If user asks to move/copy/delete/read/write a file → MUST use action: "tool", NEVER "respond" or "delegate"
2. If user asks a greeting (hi/hello) → use action: "respond"
3. If user asks code generation → use action: "delegate"

**FILE OPERATIONS - MANDATORY TOOL USAGE:**
Any request containing words like "move", "copy", "delete", "read", "write" + file path MUST return:
{{"action": "tool", "agent": null, "reasoning": "file operation", "response": null, "tool_calls": [{{"tool": "move_file", "args": {{...}}}}]}}

DO NOT suggest commands. DO NOT respond with instructions. EXECUTE the tool directly.

{tools_section}

**Examples of CORRECT behavior:**
- "copy file.py to backup.py" → {{"action": "tool", "agent": null, "reasoning": "copy file", "response": null, "tool_calls": [{{"tool": "copy_file", "args": {{"source": "file.py", "destination": "backup.py"}}}}]}}
- "move /path/to/file.py to /dest/" → {{"action": "tool", "agent": null, "reasoning": "move file", "response": null, "tool_calls": [{{"tool": "move_file", "args": {{"source": "/path/to/file.py", "destination": "/dest/file.py"}}}}]}}
- "What does app.py do?" → {{"action": "tool", "agent": null, "reasoning": "read file", "response": null, "tool_calls": [{{"tool": "read_file", "args": {{"path": "app.py"}}}}]}}
- "delete old.txt" → {{"action": "tool", "agent": null, "reasoning": "delete file", "response": null, "tool_calls": [{{"tool": "delete_file", "args": {{"path": "old.txt"}}}}]}}
- "Hello" → {{"action": "respond", "agent": null, "reasoning": "greeting", "response": "Hello! How can I help?", "tool_calls": []}}
- "Write a sorting function" → {{"action": "delegate", "agent": "{default_agent}", "reasoning": "code generation", "response": null, "tool_calls": []}}

**Agent reference format:**
- "agent:<name>" for built-in agents (example: agent:ollama, agent:codex, agent:claude_code)
- "llm:<profile>" for configured LLM profiles (example: llm:groq-70b)
- Default agent: {default_agent}

Respond with ONLY a valid JSON object:
{{
    "action": "delegate" | "respond" | "tool" | "done",
    "agent": "{default_agent}" | "agent:<name>" | "llm:<profile>" | null,
    "reasoning": "brief reason",
    "response": "only for greetings, otherwise null",
    "tool_calls": [{{"tool": "tool_name", "args": {{"param": "value"}}}}]
}}

Examples:
- "Hello" -> {{"action": "respond", "agent": null, "reasoning": "greeting", "response": "Hello! How can I help?", "tool_calls": []}}
- "copy file.py to backup.py" -> {{"action": "tool", "agent": null, "reasoning": "copy file", "response": null, "tool_calls": [{{"tool": "copy_file", "args": {{"source": "file.py", "destination": "backup.py"}}}}]}}
- "move /path/to/file.py to /dest/" -> {{"action": "tool", "agent": null, "reasoning": "move file", "response": null, "tool_calls": [{{"tool": "move_file", "args": {{"source": "/path/to/file.py", "destination": "/dest/file.py"}}}}]}}
- "delete old.txt" -> {{"action": "tool", "agent": null, "reasoning": "delete file", "response": null, "tool_calls": [{{"tool": "delete_file", "args": {{"path": "old.txt"}}}}]}}
- "What does app.py do?" -> {{"action": "tool", "agent": null, "reasoning": "read file", "response": null, "tool_calls": [{{"tool": "read_file", "args": {{"path": "app.py"}}}}]}}
- "Write a sorting function" -> {{"action": "delegate", "agent": "{default_agent}", "reasoning": "code generation", "response": null, "tool_calls": []}}
"""


# System prompt for planning (DEPRECATED - use _build_planning_system_prompt instead)
PLANNING_SYSTEM_PROMPT = """You are Codur, an autonomous coding agent orchestrator.

Your job is to analyze user requests and decide the best approach:

**For ONLY greetings (hi, hello, hey):**
- Respond with action: "respond" and a brief greeting
- Example: "Hi!" -> {"action": "respond", "response": "Hello! How can I help you?"}

**For questions about code files (what does X do, explain Y, etc):**
- Use action: "tool" to read the file first
- Example: "What does foo.py do?" -> {"action": "tool", "tool_calls": [{"tool": "read_file", "args": {"path": "foo.py"}}]}

**For code generation tasks:**
- Use action: "delegate" with an agent reference
- Example: "Write a function to sort" -> {"action": "delegate", "agent": "agent:ollama"}

**Agent reference format:**
- "agent:<name>" for built-in agents or agent profiles (example: agent:ollama or agent:local-ollama)
- "llm:<profile>" for configured LLM profiles (example: llm:openai-chatgpt-40)

**Available tools:**
See the dynamic tool list below; use tool names exactly as listed.

**IMPORTANT:**
- Questions like "what does X.py do?" need the read_file tool first
- NEVER respond directly to code questions without reading the file
- Only use action "respond" for simple greetings
- Tokens prefixed with "@" refer to file paths
- You may chain multiple tool calls in one response when needed

Respond with ONLY a valid JSON object:
{
    "action": "delegate" | "respond" | "tool" | "done",
    "agent": "ollama" | "codex" | "sheets" | "linkedin" | "agent:<name>" | "llm:<profile>" | null,
    "reasoning": "brief reason",
    "response": "only for greetings, otherwise null",
    "tool_calls": [{"tool": "read_file", "args": {"path": "..."}}]
}

Examples:
- "Hello" -> {"action": "respond", "agent": null, "reasoning": "greeting", "response": "Hello! How can I help?", "tool_calls": []}
- "What does app.py do?" -> {"action": "tool", "agent": null, "reasoning": "need to read file", "response": null, "tool_calls": [{"tool": "read_file", "args": {"path": "app.py"}}]}
- "Write a function" -> {"action": "delegate", "agent": "agent:ollama", "reasoning": "code generation", "response": null, "tool_calls": []}
- "Move file into archive dir" -> {"action": "tool", "agent": null, "reasoning": "move file", "response": null, "tool_calls": [{"tool": "move_file_to_dir", "args": {"source": "reports/today.txt", "destination_dir": "reports/archive"}}]}
- "Find and copy" -> {"action": "tool", "agent": null, "reasoning": "locate file then copy", "response": null, "tool_calls": [{"tool": "search_files", "args": {"query": "report.txt"}}, {"tool": "copy_file_to_dir", "args": {"source": "reports/report.txt", "destination_dir": "reports/archive"}}]}
"""

# Constants
DEBUG_TRUNCATE_SHORT = 500
DEBUG_TRUNCATE_LONG = 1000
def _create_llm_debug(
    system_prompt: str,
    user_message: str,
    llm_response: str,
    retry_response: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    config: Optional[CodurConfig] = None
) -> Dict[str, Any]:
    """Create debug information for LLM interactions.

    Args:
        system_prompt: System prompt sent to LLM
        user_message: User message content
        llm_response: LLM's response
        retry_response: Optional retry response if first attempt failed
        llm: Optional LLM instance for extracting model info
        config: Optional config for extracting profile info

    Returns:
        Dictionary with debug information
    """
    debug_info = {
        "node": "plan",
        "system_prompt": system_prompt[:DEBUG_TRUNCATE_SHORT] + "..." if len(system_prompt) > DEBUG_TRUNCATE_SHORT else system_prompt,
        "user_message": user_message,
        "llm_response": llm_response[:DEBUG_TRUNCATE_LONG] + "..." if len(llm_response) > DEBUG_TRUNCATE_LONG else llm_response,
    }

    # Add LLM model information
    if llm:
        model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', None) or "unknown"
        debug_info["llm_model"] = model_name

    if config:
        profile_name = config.llm.default_profile
        debug_info["llm_profile"] = profile_name

    if retry_response:
        debug_info["llm_response_retry"] = retry_response[:DEBUG_TRUNCATE_LONG] + "..." if len(retry_response) > DEBUG_TRUNCATE_LONG else retry_response
    return debug_info


def _parse_json_from_content(content: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM response content.

    Args:
        content: LLM response content that may contain JSON

    Returns:
        Parsed JSON dictionary, or None if parsing fails
    """
    # Clean the JSON response first (removes artifacts)
    cleaned = clean_json_response(content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try non-greedy regex matching (old behavior)
        json_match = re.search(r"\{.*?\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None


def _build_planning_prompts(
    messages: list[BaseMessage],
    has_tool_results: bool,
    config: CodurConfig
) -> list[BaseMessage]:
    """Build the prompt messages for planning LLM call.

    Args:
        messages: Normalized conversation messages
        has_tool_results: Whether tool results are present in messages
        config: Codur configuration for dynamic prompt generation

    Returns:
        List of messages to send to LLM
    """
    tool_lines = []
    for item in list_tool_directory():
        name = item.get("name", "")
        signature = item.get("signature", "")
        summary = item.get("summary", "")
        if not name:
            continue
        if summary:
            tool_lines.append(f"- {name}{signature}: {summary}")
        else:
            tool_lines.append(f"- {name}{signature}")
    tools_text = "\n".join(tool_lines) if tool_lines else "- (no tools available)"
    # Use dynamic prompt that injects configured default agent and available tools
    dynamic_prompt = _build_planning_system_prompt(config)
    system_prompt = SystemMessage(
        content=f"{dynamic_prompt}\n\nDynamic tools list:\n{tools_text}"
    )
    prompt_messages = [system_prompt] + list(messages)

    if has_tool_results:
        followup_prompt = SystemMessage(
            content=(
                "Tool results are available. Use them to answer the user's latest request. "
                "Do not echo large file contents; summarize what the code does. "
                "Respond with the required JSON format."
            )
        )
        prompt_messages.insert(1, followup_prompt)

    return prompt_messages


def _handle_planning_decision(
    decision: PlanningDecision,
    iterations: int,
    llm_debug: Dict[str, Any],
    config: CodurConfig
) -> PlanNodeResult:
    """Process the planning decision and return appropriate node result.

    Args:
        decision: Parsed planning decision from LLM
        iterations: Current iteration count
        llm_debug: Debug information
        config: Codur configuration

    Returns:
        Plan node result dictionary
    """
    action = decision.get("action", "delegate")

    if action == "respond":
        response_text = decision.get("response")
        if not response_text:
            response_text = decision.get("reasoning", "Task completed")
        return {
            "next_action": "end",
            "final_response": response_text,
            "iterations": iterations + 1,
            "llm_debug": llm_debug,
        }
    if action == "tool":
        return {
            "next_action": "tool",
            "tool_calls": decision.get("tool_calls", []),
            "iterations": iterations + 1,
            "llm_debug": llm_debug,
        }
    if action == "delegate":
        if not config.agents.preferences.default_agent:
            raise ValueError("agents.preferences.default_agent must be configured")
        default_agent = config.agents.preferences.default_agent
        return {
            "next_action": "delegate",
            "selected_agent": decision.get("agent", default_agent),
            "iterations": iterations + 1,
            "llm_debug": llm_debug,
        }

    # Done or unknown action
    return {
        "next_action": "end",
        "final_response": "Task completed",
        "iterations": iterations + 1,
        "llm_debug": llm_debug,
    }


def _parse_planning_response(
    llm: BaseChatModel,
    content: str,
    messages: list[BaseMessage],
    has_tool_results: bool,
    llm_debug: Dict[str, Any]
) -> Optional[PlanningDecision]:
    """Parse the LLM's planning response, with retry logic.

    Args:
        llm: Language model instance
        content: LLM response content
        messages: Conversation messages
        has_tool_results: Whether tool results are present
        llm_debug: Debug information dictionary (modified in place)

    Returns:
        Parsed planning decision, or None if all parsing attempts fail
    """
    decision = _parse_json_from_content(content)
    if decision:
        return decision

    # If no JSON found and we have tool results, try a retry
    if has_tool_results:
        retry_prompt = SystemMessage(
            content=(
                "You must return ONLY a valid JSON object. "
                "If tools are needed next, use action 'tool' with tool_calls. "
                "If you can answer now, use action 'respond' with a concise response."
            )
        )
        retry_response = llm.invoke([retry_prompt] + list(messages))
        retry_content = retry_response.content
        llm_debug["llm_response_retry"] = retry_content[:DEBUG_TRUNCATE_LONG] + "..." if len(retry_content) > DEBUG_TRUNCATE_LONG else retry_content

        decision = _parse_json_from_content(retry_content)
        if decision:
            return decision

    return None


def plan_node(state: AgentState, llm: BaseChatModel, config: CodurConfig) -> PlanNodeResult:
    """Planning node: Analyze the task and decide what to do.

    This node is the main entry point for task planning. It:
    1. Checks for trivial responses (greetings, thanks)
    2. Detects file explanation requests and auto-reads files
    3. Invokes LLM for complex decision-making
    4. Returns appropriate action (delegate, tool, or respond)

    Args:
        state: Current agent state with messages and iteration count
        llm: Language model instance for planning decisions
        config: Codur configuration (currently unused but kept for compatibility)

    Returns:
        Dictionary with next_action and relevant data (tool_calls, selected_agent, etc.)
    """
    if "config" not in state:
        raise ValueError("AgentState must include config")
    if state.get("verbose"):
        console.print("[bold blue]Planning...[/bold blue]")

    messages = _normalize_messages(state.get("messages"))
    iterations = state.get("iterations", 0)

    # Check if tool results are present in the conversation
    tool_results_present = any(
        isinstance(msg, SystemMessage) and msg.content.startswith("Tool results:")
        for msg in messages
    )

    if config.runtime.detect_tool_calls_from_text:
        non_llm_result = run_non_llm_tools(messages, state)
        if non_llm_result:
            return non_llm_result

    # Build prompt messages for LLM
    prompt_messages = _build_planning_prompts(messages, tool_results_present, config)

    # Invoke LLM to get planning decision
    response = llm.invoke(prompt_messages)
    content = response.content

    # Create debug information with dynamic prompt and LLM model info
    user_message = messages[-1].content if messages else ""
    planning_prompt = _build_planning_system_prompt(config)
    llm_debug = _create_llm_debug(planning_prompt, user_message, content, llm=llm, config=config)

    # Parse the LLM response
    try:
        decision = _parse_planning_response(llm, content, messages, tool_results_present, llm_debug)

        if decision is None:
            # If parsing failed completely, return the raw content or fallback
            if tool_results_present:
                return {
                    "next_action": "end",
                    "final_response": content,
                    "iterations": iterations + 1,
                    "llm_debug": llm_debug,
                }
            # Fallback: assume it's a coding task, use configured default agent
            default_agent = config.agents.preferences.default_agent or "agent:ollama"
            decision = {
                "action": "delegate",
                "agent": default_agent,
                "reasoning": "No clear decision",
                "response": None
            }

        # Handle the decision and return appropriate result
        return _handle_planning_decision(decision, iterations, llm_debug, config)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # More prominent error reporting
        console.print("\n" + "="*80, style="red bold")
        console.print("  PLANNING ERROR - Failed to parse LLM decision", style="red bold on yellow")
        console.print("="*80, style="red bold")
        console.print(f"  Error Type: {type(e).__name__}", style="red")
        console.print(f"  Error Message: {str(e)}", style="red")
        console.print(f"  LLM Response: {content[:200]}...", style="yellow")
        console.print("="*80 + "\n", style="red bold")

        # Add error details to debug info for TUI visibility
        llm_debug["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "raw_response": content[:500]
        }

        # Fallback: delegate to configured default agent
        default_agent = config.agents.preferences.default_agent or "agent:ollama"
        console.print(f"  Falling back to default agent: {default_agent}", style="yellow")

        return {
            "next_action": "delegate",
            "selected_agent": default_agent,
            "iterations": iterations + 1,
            "llm_debug": llm_debug,
        }
