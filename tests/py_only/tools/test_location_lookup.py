import pytest

from codur.tools import webrequests


class _FakeResponse:
    def __init__(self, url, payload):
        self._url = url
        self._payload = payload

    @property
    def url(self):
        return self._url

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_location_lookup_ipapi_with_ip(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse(
            url,
            {
                "ip": "8.8.8.8",
                "city": "Mountain View",
                "region": "California",
                "region_code": "CA",
                "country_name": "United States",
                "country_code": "US",
                "latitude": 37.4056,
                "longitude": -122.0775,
                "postal": "94043",
                "timezone": "America/Los_Angeles",
                "org": "Google LLC",
                "asn": "AS15169",
            },
        )

    monkeypatch.setattr(webrequests.requests, "get", fake_get)

    result = webrequests.location_lookup("8.8.8.8", provider="ipapi", timeout_s=5)
    assert captured["url"] == "https://ipapi.co/8.8.8.8/json/"
    assert captured["timeout"] == 5
    assert result["ip"] == "8.8.8.8"
    assert result["provider"] == "ipapi"


def test_location_lookup_ipapi_without_ip(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        return _FakeResponse(url, {"ip": "1.2.3.4"})

    monkeypatch.setattr(webrequests.requests, "get", fake_get)

    result = webrequests.location_lookup(provider="ipapi")
    assert captured["url"] == "https://ipapi.co/json/"
    assert result["ip"] == "1.2.3.4"


def test_location_lookup_invalid_provider():
    with pytest.raises(ValueError):
        webrequests.location_lookup(provider="unknown")
