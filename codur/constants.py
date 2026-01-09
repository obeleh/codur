"""
Shared constants for Codur.
"""

from enum import Enum


class TaskType(Enum):
    """Task type classification for routing."""

    GREETING = "greeting"
    FILE_OPERATION = "file_operation"
    CODE_FIX = "code_fix"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    RESULT_VERIFICATION = "result_verification"  # For verification agent response tools (build_verification_response)
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"
    REFACTOR = "complex"
    WEB_SEARCH = "web_search"
    UNKNOWN = "unknown"
    CODE_ANALYSIS = "code_analysis"
    META_TOOL = "meta_tool" # Tools used to implement (agentic) logic or to communicate with other agents

# General constants
GREETING_MAX_WORDS = 3
DEFAULT_MAX_BYTES = 200_000
DEFAULT_MAX_RESULTS = 200
DEFAULT_CLI_TIMEOUT = 600
DEFAULT_MAX_ITERATIONS = 10

# Agent identifiers (without prefix)
AGENT_CODING = "codur-coding"
AGENT_EXPLAINING = "codur-explaining"

# Routing actions
ACTION_DELEGATE = "delegate"
ACTION_TOOL = "tool"
ACTION_RESPOND = "respond"
ACTION_END = "end"
ACTION_CONTINUE = "continue"
ACTION_VERIFICATION_AGENT = "verification"
ACTION_CODING = "coding"

# Agent references (with prefix)
PREFIX_AGENT = "agent:"
REF_AGENT_CODING = f"{PREFIX_AGENT}{AGENT_CODING}"
REF_AGENT_EXPLAINING = f"{PREFIX_AGENT}{AGENT_EXPLAINING}"
