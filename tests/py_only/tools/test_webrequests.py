"""Tests for web request tools."""

import pytest

from codur.tools import webrequests
from codur.tools.webrequests import fetch_webpage


class _FakeResponse:
    def __init__(self, text: str, content_type: str = "text/plain", status_code: int = 200):
        self.text = text
        self.status_code = status_code
        self.ok = status_code < 400
        self.headers = {"content-type": content_type}
        self.url = "https://example.com"
        self.apparent_encoding = "utf-8"
        self.encoding = None


def test_fetch_webpage_plain_text(monkeypatch):
    def fake_request(*args, **kwargs):
        return _FakeResponse("plain text", content_type="text/plain")

    monkeypatch.setattr(webrequests.requests, "request", fake_request)

    result = fetch_webpage(
        "https://example.com",
        output_format="text",
        cleanup_level="none",
        include_html=True,
    )
    assert result["ok"] is True
    assert result["format"] == "text"
    assert result["text"] == "plain text"
    assert result["html"] == "plain text"


def test_fetch_webpage_html_text(monkeypatch):
    html = "<html><body><p>Hello</p></body></html>"

    def fake_request(*args, **kwargs):
        return _FakeResponse(html, content_type="text/html")

    monkeypatch.setattr(webrequests.requests, "request", fake_request)

    result = fetch_webpage(
        "https://example.com",
        output_format="text",
        cleanup_level="none",
    )
    assert result["ok"] is True
    assert "Hello" in result["text"]


def test_fetch_webpage_rejects_invalid_output_format():
    def fake_request(*args, **kwargs):
        return _FakeResponse("plain text", content_type="text/plain")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(webrequests.requests, "request", fake_request)
        with pytest.raises(ValueError, match="output_format must be one of"):
            fetch_webpage(
                "https://example.com",
                output_format="pdf",
                cleanup_level="none",
                clean=False,
            )
