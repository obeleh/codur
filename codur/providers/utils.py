"""
Utilities for providers.
"""

import importlib
from typing import Any


def lazy_import(module_name: str, install_msg: str) -> Any:
    """Import a module lazily or raise a helpful error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(install_msg) from exc
