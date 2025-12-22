"""
Web request tools for Codur.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

import requests

from codur.constants import DEFAULT_MAX_BYTES
from codur.graph.state import AgentState

try:
    from readability import Document
except ImportError:  # pragma: no cover - optional dependency
    Document = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None


def _truncate(text: str, max_bytes: int) -> str:
    if len(text) <= max_bytes:
        return text
    return text[:max_bytes] + "\n... [truncated]"


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _basic_title(html: str) -> str:
    if BeautifulSoup is None:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""


def _extract_text_from_html(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text("\n")
    else:
        text = re.sub(r"<[^>]+>", " ", html)
    return _collapse_whitespace(unescape(text))


def _extract_readability(html: str) -> dict[str, str]:
    if Document is None:
        raise RuntimeError("readability-lxml is not installed")
    doc = Document(html)
    content_html = doc.summary(html_partial=True)
    return {
        "title": doc.title() or "",
        "text": _extract_text_from_html(content_html),
        "html": content_html,
        "extractor": "readability",
    }


def _extract_basic(html: str) -> dict[str, str]:
    return {
        "title": _basic_title(html),
        "text": _extract_text_from_html(html),
        "html": html,
        "extractor": "basic",
    }


def _extract_main(html: str, mode: str) -> dict[str, str]:
    normalized = (mode or "auto").lower().strip()
    if normalized == "readability":
        if Document is None:
            raise RuntimeError("readability-lxml is not installed")
        return _extract_readability(html)
    if normalized == "basic":
        return _extract_basic(html)
    if normalized == "auto":
        if Document is not None:
            try:
                return _extract_readability(html)
            except Exception:
                pass
        return _extract_basic(html)
    raise ValueError("extract_mode must be one of: auto, readability, basic")


def fetch_webpage(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | str | None = None,
    timeout_s: float = 20.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    clean: bool = True,
    extract_mode: str = "auto",
    include_html: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Fetch a webpage and optionally extract the main text content.
    """
    merged_headers = {"User-Agent": "codur/1.0"}
    if headers:
        merged_headers.update(headers)
    response = requests.request(
        method=method,
        url=url,
        headers=merged_headers,
        params=params,
        data=data,
        timeout=timeout_s,
    )
    response.encoding = response.apparent_encoding or response.encoding
    content_type = response.headers.get("content-type", "")
    html = response.text

    title = ""
    extractor = "raw"
    text = html
    html_output = html
    if clean and "html" in content_type.lower():
        extracted = _extract_main(html, extract_mode)
        title = extracted.get("title", "")
        extractor = extracted.get("extractor", "auto")
        text = extracted.get("text", "")
        html_output = extracted.get("html", html)

    text = _truncate(text, max_bytes)
    result = {
        "url": response.url,
        "status_code": response.status_code,
        "ok": response.ok,
        "content_type": content_type,
        "title": title,
        "extractor": extractor,
        "text": text,
    }

    if include_html:
        result["html"] = _truncate(html_output, max_bytes)

    return result
