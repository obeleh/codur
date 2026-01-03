from langchain_core.messages import BaseMessage, SystemMessage

from codur.graph.state import AgentState
from codur.graph.state_operations import get_config
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
"""


def summarize_message_history(messages: list[BaseMessage], state: AgentState) -> str:
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

    response = invoke_llm(
        llm,
         messages + [system_message],
        invoked_by="summarizer",
        state=state,
        config=config,
    )

    return response.content