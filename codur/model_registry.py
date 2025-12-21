"""
Model listing helpers for LLM providers.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import requests


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def list_groq_models(api_key: str | None = None) -> list[str]:
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set")
    url = "https://api.groq.com/openai/v1/models"
    response = requests.get(url, headers=_auth_headers(key), timeout=30)
    response.raise_for_status()
    data = response.json()
    return sorted([item["id"] for item in data.get("data", []) if "id" in item])


def list_openai_models(
    api_key: str | None = None,
    months: int = 15,
) -> list[dict[str, Any]]:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")
    url = "https://api.openai.com/v1/models"
    response = requests.get(url, headers=_auth_headers(key), timeout=30)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])

    cutoff = datetime.now() - timedelta(days=months * 30)
    cutoff_timestamp = int(cutoff.timestamp())

    filtered = []
    for model in models:
        model_id = model.get("id", "")
        created = model.get("created", 0)

        if not model_id.startswith(("gpt-", "o1", "o3")):
            continue

        if created and created < cutoff_timestamp:
            continue

        if any(char.isdigit() for char in model_id.split("-")[-1]) and len(model_id.split("-")) > 2:
            continue

        skip_keywords = [
            "audio", "whisper", "tts",
            "realtime", "preview",
            "search", "transcribe", "diarize",
            "codex", "image",
            "latest", "chat-latest",
            "api",
        ]
        if any(keyword in model_id.lower() for keyword in skip_keywords):
            continue

        filtered.append(model)

    return sorted(filtered, key=lambda item: item.get("id", ""))


def list_anthropic_models(api_key: str | None = None) -> list[str]:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    return sorted([item.get("id") for item in data.get("data", []) if item.get("id")])


def list_ollama_models(base_url: str | None = None) -> list[str]:
    url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    response = requests.get(f"{url}/api/tags", timeout=10)
    response.raise_for_status()
    data = response.json()
    return sorted([item.get("name") for item in data.get("models", []) if item.get("name")])


def list_ollama_registry_models(
    limit: int = 200,
    sort: str = "top",
    max_size_gb: float | None = None,
) -> list[str]:
    url = "https://registry.ollama.ai/api/tags"
    params = {"limit": limit, "sort": sort}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    models = []
    max_bytes = None
    if max_size_gb is not None:
        max_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    for item in data.get("models", []):
        name = item.get("name")
        size = item.get("size")
        if not name:
            continue
        if max_bytes is not None and isinstance(size, int) and size > max_bytes:
            continue
        models.append(name)
    return sorted(models)
