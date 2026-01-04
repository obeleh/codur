from langchain_core.messages import SystemMessage, BaseMessage

from codur.graph.state import AgentState
from codur.graph.state_operations import get_config, get_messages, get_tool_calls_parsed, ToolOutput, \
    get_parsed_tool_calls_from_messages
from codur.llm import create_llm_profile
from codur.tools.registry import get_tool_summary_format
from codur.utils.llm_calls import invoke_llm

PROMPT = """
You are an expert at summarizing conversation history for AI agents. Given the following conversation messages between a user and an AI agent, produce a concise summary that captures the key points, decisions, and actions taken. The summary should be clear and informative, allowing someone who hasn't seen the full conversation to understand what transpired.
When summarizing, focus on:
- Important tasks or goals discussed
- Actions taken by the agent and their outputs

It is not your job to execute the humans requests, only to summarize what has happened in the conversation so far.
It is very important to include the tool results in the summary. But please follow the format specified below.
Do not call any tools yourself!

Tool result format:
"""


def get_messages_to_summarize(state: AgentState) -> list[BaseMessage]:
    messages = get_messages(state)
    last_system_idx = 0
    for idx, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            last_system_idx = idx
    return messages[last_system_idx:]


def get_tool_formats(messages: list[BaseMessage]) -> str:
    tool_formats = []
    for tool_msg in get_parsed_tool_calls_from_messages(messages):
        assert isinstance(tool_msg, ToolOutput)
        fmt = get_tool_summary_format(tool_msg.tool)
        fmt = f"for {tool_msg.tool}:\n{fmt}\n"
        tool_formats.append(fmt)

    return "\n".join(tool_formats)


def create_summary(state: AgentState) -> str:
    config = get_config(state)

    llm = create_llm_profile(
        config,
        config.llm.default_profile,
        json_mode=False,
        temperature=0.0,
    )

    messages = get_messages_to_summarize(state)
    tool_formats = get_tool_formats(messages)

    system_message = SystemMessage(
        content=PROMPT + "\n\n" + tool_formats
    )

    response = invoke_llm(
        llm,
        messages + [system_message],
        invoked_by="summarizer",
        state=state,
        config=config,
    )

    return response.content


def prepend_summary(func):
    def inner(state: AgentState, *args, **kwargs):
        if "summary" in kwargs:
            return func(state, *args, **kwargs)
        else:
            summary_result = create_summary(state)
            kwargs["summary"] = summary_result
            agent_result = func(state, *args, **kwargs)
            agent_result["agent_summaries"] = [summary_result]
            return agent_result

    return inner
