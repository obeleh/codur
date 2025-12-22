"""
DuckDuckGo search tools for Codur.
"""

from __future__ import annotations

import inspect

from codur.graph.state import AgentState

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - optional dependency
    DDGS = None


def duckduckgo_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    backend: str | None = None,
    state: AgentState | None = None,
) -> list[dict]:
    """
    Run a DuckDuckGo search and return text results.
    """
    if DDGS is None:
        raise RuntimeError("duckduckgo-search is not installed")

    with DDGS() as ddgs:
        kwargs = {
            "region": region,
            "safesearch": safesearch,
            "timelimit": timelimit,
            "max_results": max_results,
        }
        if backend:
            if "backend" in inspect.signature(ddgs.text).parameters:
                kwargs["backend"] = backend
        results = list(ddgs.text(query, **kwargs))
    return results
