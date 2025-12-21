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


class JSONResponseParser:
    def parse(self, content: str) -> Optional[Dict[str, Any]]:
        cleaned = clean_json_response(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*?\}", content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return None
            return None
