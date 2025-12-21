#!/usr/bin/env python3
"""
Standalone launcher for Codur TUI

Usage:
    python -m codur.tui_standalone
    # or
    python codur/tui_standalone.py
"""

from codur.tui import run_tui
from codur.config import load_config

if __name__ == "__main__":
    config = load_config()
    run_tui(config)
