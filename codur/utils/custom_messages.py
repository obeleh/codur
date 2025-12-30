from langchain_core.messages import SystemMessage


class ShortenableSystemMessage(SystemMessage):
    """System message with a shorter summary for LLM calls."""

    short_content: str | None = None
    long_form_visible_for_agent_name: str | None = None
    exact_agent_name: str | None = None