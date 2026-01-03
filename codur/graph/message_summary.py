from langchain_core.messages import SystemMessage, BaseMessage

from codur.graph.state import AgentState
from codur.graph.state_operations import get_config, get_messages
from codur.llm import create_llm_profile
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

<tool_name>: <tool args>
```
<tool_result_content>
```

For file tool_calls:
<file_name>:
```
<file_content>
```

For python_ast_dependencies
```
a -> b
b -> c
```
"""


def get_messages_to_summarize(state: AgentState) -> list[BaseMessage]:
    messages = get_messages(state)
    last_system_idx = 0
    for idx, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            last_system_idx = idx

    return messages[last_system_idx:]


def create_summary(state: AgentState) -> str:
    config = get_config(state)

    llm = create_llm_profile(
        config,
        config.llm.default_profile,
        json_mode=False,
        temperature=0.0,
    )

    system_message = SystemMessage(
        content=PROMPT
    )

    messages = get_messages_to_summarize(state)

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
