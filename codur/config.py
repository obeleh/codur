"""
Configuration management for Codur
"""

import os
from dotenv import load_dotenv
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

DEFAULT_CONFIG_RELATIVE_PATH = Path(__file__).resolve().parent.parent / "codur.yaml"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server"""
    command: str
    args: List[str] = Field(default_factory=list)
    cwd: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Configuration for an agent"""
    name: str
    type: str  # "mcp", "llm", "tool"
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentPreferences(BaseModel):
    """Agent routing and preferences"""
    default_agent: str = ""
    routing: Dict[str, str] = Field(default_factory=dict)
    fallback_order: List[str] = Field(default_factory=list)


class AgentSettings(BaseModel):
    """Overall agent settings"""
    preferences: AgentPreferences = Field(default_factory=AgentPreferences)
    configs: Dict[str, AgentConfig] = Field(default_factory=dict)
    profiles: Dict[str, AgentConfig] = Field(default_factory=dict)


class LLMProfile(BaseModel):
    """Named LLM profile for multi-provider usage."""
    provider: str
    model: str
    temperature: Optional[float] = None
    api_key_env: Optional[str] = None


class LLMProviderSettings(BaseModel):
    """Provider-specific settings for LLMs."""
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    max_model_size_gb: Optional[float] = None


class LLMSettings(BaseModel):
    """LLM provider settings"""
    default_profile: str
    profiles: Dict[str, LLMProfile] = Field(default_factory=dict)
    default_temperature: float = 0.7
    # Runtime API keys (loaded from environment)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


class AsyncSettings(BaseModel):
    """Async execution settings"""
    max_concurrent_agents: int = 3
    event_buffer_size: int = 200
    stream_events: bool = True
    user_input_poll_ms: int = 100
    tool_timeout_s: int = 600
    cancel_grace_s: int = 5


class RuntimeSettings(BaseModel):
    """Runtime execution settings"""
    max_iterations: int = 10
    verbose: bool = False
    allow_outside_workspace: bool = False
    detect_tool_calls_from_text: bool = True
    planner_fallback_profiles: List[str] = Field(default_factory=list)
    async_: AsyncSettings = Field(default_factory=AsyncSettings, alias="async")

    class Config:
        populate_by_name = True


class CodurConfig(BaseModel):
    """Main Codur configuration"""
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    providers: Dict[str, LLMProviderSettings] = Field(default_factory=dict)

    # Convenience properties for existing code that expects flat fields
    @property
    def llm_provider(self) -> str:
        return self.llm.default_profile or ""

    @property
    def llm_model(self) -> str:
        return ""

    @property
    def llm_temperature(self) -> float:
        return 0.0

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return self.llm.anthropic_api_key

    @property
    def openai_api_key(self) -> Optional[str]:
        return self.llm.openai_api_key

    @property
    def groq_api_key(self) -> Optional[str]:
        return self.llm.groq_api_key

    @property
    def verbose(self) -> bool:
        return self.runtime.verbose

    @property
    def max_iterations(self) -> int:
        return self.runtime.max_iterations


def load_config(config_path: Optional[Path] = None) -> CodurConfig:
    """
    Load configuration from YAML file or use defaults.

    Priority:
    1. Provided config_path
    2. ./codur.yaml
    3. ~/.codur/config.yaml
    4. Claude Desktop config (for MCP servers)
    5. Defaults
    """
    config_data = {}

    # Load environment variables from .env (if present)
    load_dotenv()

    # Search for config files
    search_paths = []
    if config_path:
        search_paths.append(config_path)

    search_paths.extend([
        DEFAULT_CONFIG_RELATIVE_PATH,
        Path.home() / ".codur" / "config.yaml",
    ])

    for path in search_paths:
        if path.exists():
            with open(path, "r") as f:
                if path.suffix in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f) or {}
                else:
                    config_data = json.load(f)
            break

    # Inject MCP servers from Claude Desktop config if missing
    if not config_data.get("mcp_servers"):
        claude_config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        if claude_config_path.exists():
            with open(claude_config_path) as f:
                claude_config = json.load(f)
            config_data["mcp_servers"] = claude_config.get("mcpServers", {})

    # Resolve API keys from environment
    llm = config_data.setdefault("llm", {})
    providers = config_data.setdefault("providers", {})

    # Back-compat: move old ollama max size into providers
    if "ollama_max_model_size_gb" in llm:
        providers.setdefault("ollama", {})
        providers["ollama"].setdefault("max_model_size_gb", llm.pop("ollama_max_model_size_gb"))

    # Back-compat: promote old top-level api_keys into providers
    legacy_api_keys = config_data.pop("api_keys", {})
    if isinstance(legacy_api_keys, dict):
        for key, value in legacy_api_keys.items():
            if key.endswith("_env"):
                provider = key[:-4]
                providers.setdefault(provider, {})
                providers[provider].setdefault("api_key_env", value)

    legacy_llm_providers = llm.pop("providers", {})
    if isinstance(legacy_llm_providers, dict):
        for provider, value in legacy_llm_providers.items():
            providers.setdefault(provider, {})
            if isinstance(value, dict):
                for option, option_value in value.items():
                    providers[provider].setdefault(option, option_value)

    def _provider_env(provider: str, fallback: str) -> str:
        provider_cfg = providers.get(provider, {}) if isinstance(providers, dict) else {}
        return provider_cfg.get("api_key_env") or fallback

    anthropic_env = _provider_env("anthropic", "ANTHROPIC_API_KEY")
    openai_env = _provider_env("openai", "OPENAI_API_KEY")
    groq_env = _provider_env("groq", "GROQ_API_KEY")

    llm["anthropic_api_key"] = os.getenv(anthropic_env)
    llm["openai_api_key"] = os.getenv(openai_env)
    llm["groq_api_key"] = os.getenv(groq_env)

    # Back-compat: if old flat keys exist, map to default_profile if possible
    if "llm_provider" in config_data or "llm_model" in config_data:
        llm.setdefault("default_profile", None)

    return CodurConfig(**config_data)


def save_config(config: CodurConfig, path: Optional[Path] = None):
    """Save configuration to YAML file."""
    if path is None:
        path = Path.home() / ".codur" / "config.yaml"

    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, excluding sensitive data
    config_dict = config.model_dump(
        exclude={
            "llm": {"anthropic_api_key", "openai_api_key", "groq_api_key"}
        }
    )

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
