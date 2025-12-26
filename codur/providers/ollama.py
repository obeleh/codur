"""
Ollama LLM provider implementation.
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from codur.config import CodurConfig
from codur.providers.base import BaseLLMProvider, ProviderRegistry
from codur.providers.utils import lazy_import


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
        langchain_ollama = lazy_import(
            "langchain_ollama",
            "Ollama support requires langchain-ollama. Install with `pip install langchain-ollama`."
        )
        ChatOllama = langchain_ollama.ChatOllama

        # Resolve base URL from config
        base_url = "http://localhost:11434"
        ollama_cfg = config.agents.configs.get("ollama")
        if ollama_cfg and hasattr(ollama_cfg, "config") and "base_url" in ollama_cfg.config:
            base_url = ollama_cfg.config["base_url"]
        
        # Check provider specific config
        ollama_provider = config.providers.get("ollama")
        if ollama_provider and ollama_provider.base_url:
            base_url = ollama_provider.base_url

        # Check MCP server config as fallback (logic matched from OllamaAgent)
        ollama_mcp = config.mcp_servers.get("ollama", {})
        if hasattr(ollama_mcp, "env") and "OLLAMA_HOST" in ollama_mcp.env:
            base_url = ollama_mcp.env["OLLAMA_HOST"]

        kwargs = {
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
        }

        if json_mode:
            kwargs["format"] = "json"

        return ChatOllama(**kwargs)

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
