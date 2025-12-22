"""Lightweight token estimation for prompt budgeting."""

from __future__ import annotations

from typing import List
from langchain_core.messages import BaseMessage


class TokenCounter:
    CHARS_PER_TOKEN = 4
    MESSAGE_OVERHEAD = 10

    @staticmethod
    def count_messages(messages: List[BaseMessage]) -> int:
        total = 0
        for msg in messages:
            total += TokenCounter.MESSAGE_OVERHEAD
            total += len(msg.content) // TokenCounter.CHARS_PER_TOKEN
        return total

    @staticmethod
    def count_text(text: str) -> int:
        return len(text) // TokenCounter.CHARS_PER_TOKEN

    @classmethod
    def summarize_message(cls, msg: BaseMessage, max_tokens: int) -> BaseMessage:
        if cls.count_messages([msg]) < max_tokens:
            return msg

        content = msg.content
        lines = content.split("\n")
        kept_lines = max(1, int(max_tokens / 5))
        summary = "\n".join(lines[:kept_lines]) + "\n...(truncated)\n" + "\n".join(lines[-5:])
        msg.content = summary
        return msg
