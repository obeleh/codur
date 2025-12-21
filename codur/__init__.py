"""
Codur - Autonomous Coding Agent Orchestrator

A LangGraph-based agent that orchestrates coding tasks by delegating to:
- MCP servers (Ollama, Sheets, LinkedIn, etc.)
- Specialized coding agents (Codex, etc.)
- Its own reasoning capabilities
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
    module="langchain_core._api.deprecation",
)

__version__ = "0.1.0"
