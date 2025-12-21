"""
Ollama LLM provider implementation.
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from codur.config import CodurConfig
from codur.agents.ollama_langchain import ChatOllama
from codur.providers.base import BaseLLMProvider, ProviderRegistry


class OllamaProvider(BaseLLMProvider):
    """Ollama provider implementation."""

    @staticmethod
    def create(
        config: CodurConfig,
        model: str,
        temperature: float,
        api_key: Optional[str] = None,
        json_mode: bool = False
    ) -> BaseChatModel:
        """Create Ollama chat model.

        Args:
            config: Codur configuration
            model: Model name
            temperature: Generation temperature
            api_key: Unused for Ollama (local models)
            json_mode: If True, force JSON output format

        Returns:
            BaseChatModel: Configured ChatOllama instance
        """
        # Update config with the requested model
        data = config.model_dump()
        ollama_cfg = data.get("agents", {}).get("configs", {}).get("ollama")
        if ollama_cfg and isinstance(ollama_cfg, dict):
            ollama_cfg.setdefault("config", {})["model"] = model
        updated = CodurConfig(**data)
        return ChatOllama(config=updated, json_mode=json_mode)

    @staticmethod
    def get_api_key_env_name(config: CodurConfig) -> str:
        """Return API key environment variable name.

        Args:
            config: Codur configuration

        Returns:
            str: Empty string (Ollama doesn't use API keys)
        """
        return ""  # Ollama is local, no API key needed

    @staticmethod
    def provider_name() -> str:
        """Return provider name.

        Returns:
            str: "ollama"
        """
        return "ollama"


# Register the provider
ProviderRegistry.register("ollama", OllamaProvider)
