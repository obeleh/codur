"""JSON parsing helpers for planning responses."""

from __future__ import annotations

import json
import re
from typing import Optional, Dict, Any


def clean_json_response(response: str) -> str:
    """Extract the JSON object from a response string."""
    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace == -1 or last_brace == -1:
        return response.strip()
    return response[first_brace:last_brace + 1]


def extract_json_from_markdown(response: str) -> str:
    """Extract JSON from markdown code blocks.

    Handles markdown-wrapped JSON in these formats:
    - ```json\n{...}\n```
    - ```\n{...}\n```

    Args:
        response: Response that may contain markdown code blocks

    Returns:
        str: Extracted JSON or original response if no markdown found
    """
    if "```json" in response:
        try:
            return response.split("```json")[1].split("```")[0].strip()
        except (IndexError, AttributeError):
            pass

    if "```" in response:
        try:
            return response.split("```")[1].split("```")[0].strip()
        except (IndexError, AttributeError):
            pass

    return response


class JSONResponseParser:
    """Parser for JSON responses from LLMs, with support for markdown code blocks."""

    def parse(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response content.

        Tries multiple strategies:
        1. Extract from markdown code blocks
        2. Clean and parse JSON
        3. Regex-based JSON extraction

        Args:
            content: Response content containing JSON

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON object, or None if parsing fails
        """
        # First try markdown extraction
        markdown_extracted = extract_json_from_markdown(content)

        # Then try standard cleaning
        cleaned = clean_json_response(markdown_extracted)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback to regex-based extraction
            json_match = re.search(r"\{.*?\}", markdown_extracted, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return None
            return None
