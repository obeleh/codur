"""
Shared constants for Codur.
"""

# General constants
DEBUG_TRUNCATE_SHORT = 500
DEBUG_TRUNCATE_LONG = 1000
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

# Agent references (with prefix)
PREFIX_AGENT = "agent:"
REF_AGENT_CODING = f"{PREFIX_AGENT}{AGENT_CODING}"
REF_AGENT_EXPLAINING = f"{PREFIX_AGENT}{AGENT_EXPLAINING}"
