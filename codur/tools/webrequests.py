"""
Web request tools for Codur.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

import requests

from codur.constants import DEFAULT_MAX_BYTES, TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import tool_scenarios
from codur.utils.text_helpers import truncate_chars

try:
    from readability import Document
except ImportError:  # pragma: no cover - optional dependency
    Document = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from markdownify import markdownify as _markdownify
except ImportError:  # pragma: no cover - optional dependency
    _markdownify = None


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


def _clean_html_basic(html: str) -> str:
    if BeautifulSoup is None:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return str(soup)


def _clean_html_serp(html: str) -> str:
    if BeautifulSoup is None:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "input",
        "button",
        "select",
        "textarea",
        "iframe",
        "canvas",
        "img",
        "picture",
        "figure",
    ]):
        tag.decompose()

    noise_keywords = {
        "nav",
        "navbar",
        "header",
        "footer",
        "sidebar",
        "cookie",
        "consent",
        "advert",
        "ads",
        "sponsor",
        "promo",
        "banner",
        "login",
        "signin",
        "sign-in",
        "signup",
        "sign-up",
        "subscribe",
        "newsletter",
        "modal",
        "popup",
        "breadcrumb",
        "filters",
        "filter",
        "pagination",
        "pager",
        "topbar",
        "toolbar",
        "searchbox",
        "search-box",
        "related",
    }
    for element in soup.find_all(True):
        classes = " ".join(element.get("class", [])).lower()
        element_id = (element.get("id") or "").lower()
        if any(keyword in classes for keyword in noise_keywords) or any(
            keyword in element_id for keyword in noise_keywords
        ):
            element.decompose()

    main = soup.find("main")
    if main is not None and main.get_text(strip=True):
        return str(main)
    return str(soup)


def _extract_text_from_html(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text("\n")
    else:
        text = re.sub(r"<[^>]+>", " ", html)
    return _collapse_whitespace(unescape(text))


def _html_to_markdown(html: str) -> str:
    if _markdownify is not None:
        return _collapse_whitespace(_markdownify(html, heading_style="ATX"))
    if BeautifulSoup is not None:
        return _extract_text_from_html(html)
    raise RuntimeError("markdownify is not installed")


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
        "html": _clean_html_basic(html),
        "extractor": "basic",
    }


def _extract_serp(html: str) -> dict[str, str]:
    cleaned = _clean_html_serp(html)
    return {
        "title": _basic_title(html),
        "text": _extract_text_from_html(cleaned),
        "html": cleaned,
        "extractor": "serp",
    }


def _extract_main(html: str, mode: str) -> dict[str, str]:
    normalized = (mode or "auto").lower().strip()
    if normalized == "readability":
        if Document is None:
            raise RuntimeError("readability-lxml is not installed")
        return _extract_readability(html)
    if normalized == "basic":
        return _extract_basic(html)
    if normalized == "serp":
        return _extract_serp(html)
    if normalized == "none":
        return {"title": _basic_title(html), "text": _extract_text_from_html(html), "html": html, "extractor": "none"}
    if normalized == "auto":
        if Document is not None:
            try:
                return _extract_readability(html)
            except Exception:
                pass
        return _extract_basic(html)
    raise ValueError("cleanup_level must be one of: auto, readability, basic, serp, none")


def _resolve_cleanup_level(
    cleanup_level: str | None,
    clean: bool,
    extract_mode: str,
) -> str:
    if cleanup_level:
        return cleanup_level
    if not clean:
        return "none"
    return extract_mode or "auto"


def _resolve_output_format(requested: str | None) -> str:
    normalized = (requested or "markdown").lower().strip()
    if normalized in {"markdown", "md"}:
        return "markdown"
    if normalized in {"text", "plain"}:
        return "text"
    raise ValueError("output_format must be one of: markdown, text")


@tool_scenarios(TaskType.WEB_SEARCH)
def fetch_webpage(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | str | None = None,
    timeout_s: float = 20.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    cleanup_level: str | None = None,
    output_format: str = "markdown",
    clean: bool = True,
    extract_mode: str = "auto",
    include_html: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Fetch a webpage and extract main content with optional cleanup.

    cleanup_level values: auto, readability, basic, serp, none.
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
    cleanup = _resolve_cleanup_level(cleanup_level, clean, extract_mode)
    output_format = _resolve_output_format(output_format)
    if "html" in content_type.lower():
        extracted = _extract_main(html, cleanup)
        title = extracted.get("title", "")
        extractor = extracted.get("extractor", "auto")
        html_output = extracted.get("html", html)
        if output_format == "markdown":
            text = _html_to_markdown(html_output)
        else:
            text = extracted.get("text", "")

    text = truncate_chars(text, max_chars=max_bytes)
    result = {
        "url": response.url,
        "status_code": response.status_code,
        "ok": response.ok,
        "content_type": content_type,
        "title": title,
        "extractor": extractor,
        "format": output_format,
        "text": text,
    }

    if include_html:
        result["html"] = truncate_chars(html_output, max_chars=max_bytes)

    return result


@tool_scenarios(TaskType.WEB_SEARCH)
def location_lookup(
    ip: str | None = None,
    provider: str = "ipapi",
    timeout_s: float = 10.0,
    state: AgentState | None = None,
) -> dict:
    """Resolve an IP address to a geographic location."""
    normalized = (provider or "ipapi").lower().strip()
    if normalized not in {"ipapi", "ip-api"}:
        raise ValueError("provider must be one of: ipapi, ip-api")

    target_ip = ip.strip() if ip else ""
    if normalized == "ipapi":
        url = f"https://ipapi.co/{target_ip}/json/" if target_ip else "https://ipapi.co/json/"
        response = requests.get(url, headers={"User-Agent": "codur/1.0"}, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
        return {
            "provider": "ipapi",
            "ip": data.get("ip") or target_ip or None,
            "city": data.get("city"),
            "region": data.get("region"),
            "region_code": data.get("region_code"),
            "country": data.get("country_name"),
            "country_code": data.get("country_code"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "postal": data.get("postal"),
            "timezone": data.get("timezone"),
            "org": data.get("org"),
            "asn": data.get("asn"),
            "raw": data,
        }

    url = f"http://ip-api.com/json/{target_ip}" if target_ip else "http://ip-api.com/json/"
    response = requests.get(url, headers={"User-Agent": "codur/1.0"}, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    return {
        "provider": "ip-api",
        "ip": data.get("query") or target_ip or None,
        "city": data.get("city"),
        "region": data.get("regionName"),
        "region_code": data.get("region"),
        "country": data.get("country"),
        "country_code": data.get("countryCode"),
        "latitude": data.get("lat"),
        "longitude": data.get("lon"),
        "postal": data.get("zip"),
        "timezone": data.get("timezone"),
        "org": data.get("org"),
        "asn": data.get("as"),
        "raw": data,
    }
