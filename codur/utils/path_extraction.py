"""
Path extraction helpers for parsing user messages.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from codur.graph.state import AgentState
from codur.utils.ignore_utils import (
    get_config_from_state,
    get_exclude_dirs,
    is_gitignored,
    is_hidden_path,
    load_gitignore,
    should_include_hidden,
    should_respect_gitignore,
)


def looks_like_path(token: str) -> bool:
    """Check if a token appears to be a file path."""
    if token.startswith("@"):
        return True
    if "/" in token or "\\" in token:
        return True
    if re.search(r"\.[A-Za-z0-9]{1,5}$", token):
        return True
    return False


def extract_path_from_message(text: str) -> Optional[str]:
    """Extract first file path from message text."""
    at_match = re.search(r"@([^\s,]+)", text)
    if at_match:
        return at_match.group(1)

    in_match = re.search(r"(?:in|inside)\s+([^\s,]+)", text, re.IGNORECASE)
    if in_match:
        candidate = in_match.group(1).strip().strip(".,:;()[]{}'\"`")
        if looks_like_path(candidate):
            return candidate.lstrip("@")
        return None

    path_match = re.search(r"([^\s,]+\.py)", text)
    if path_match:
        return path_match.group(1).strip("'\"`")

    return None


def extract_file_paths(text: str) -> list[str]:
    """Extract all file paths from text."""
    paths = []

    # @file.py syntax
    at_matches = re.findall(r"@([^\s,]+)", text)
    paths.extend(at_matches)

    ext_pattern = r"(?:json|yaml|html|css|yml|txt|py|js|ts|md)"

    # Explicit file extensions
    ext_matches = re.findall(
        rf"([^\s,\"'`]+\.{ext_pattern})",
        text,
    )
    paths.extend(ext_matches)

    # Quoted paths
    quoted = re.findall(r"([\"'`])([^\"'`]+)\1", text)
    for _, q in quoted:
        if not ("/" in q or "." in q):
            continue
        if re.search(r"\s", q):
            for token in q.split():
                cleaned = token.strip(".,:;()[]{}")
                if looks_like_path(cleaned):
                    paths.append(cleaned)
            continue
        paths.append(q)

    # Clean paths: strip leading '@' and remove duplicates
    cleaned_paths = {p.lstrip("@") for p in paths}

    return list(cleaned_paths)


def find_workspace_match(raw_text: str, state: AgentState | None = None) -> Optional[str]:
    """Find a file in workspace that matches references in text."""
    text = raw_text.strip()
    if not text:
        return None

    # Extract all potential path tokens
    candidates = []
    for token in text.replace('"', " ").replace("'", " ").split():
        if token.startswith("@") or ".py" in token or "/" in token or "\\" in token:
            cleaned = token.strip(".,:;()[]{}")
            if cleaned.startswith("@"):
                cleaned = cleaned[1:]
            if cleaned:
                candidates.append(cleaned)

    cwd = Path.cwd()
    config = get_config_from_state(state)
    exclude_dirs = get_exclude_dirs(config)
    include_hidden = should_include_hidden(config)
    gitignore_spec = load_gitignore(cwd) if should_respect_gitignore(config) else None

    # Try each candidate
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = cwd / path

        # Check if path is within workspace bounds
        try:
            if path.is_absolute() and not path.resolve().is_relative_to(cwd.resolve()):
                continue
        except (ValueError, OSError):
            continue

        # Direct path match
        if path.exists():
            try:
                return str(path.relative_to(cwd) if path.is_relative_to(cwd) else path)
            except ValueError:
                continue

        # Search workspace for filename
        if path.name == candidate and not ("/" in candidate or "\\" in candidate):
            matches = []
            for root, dirnames, filenames in os.walk(cwd):
                rel_dir = Path(root).relative_to(cwd)
                filtered_dirs: list[str] = []
                for dirname in dirnames:
                    if dirname in exclude_dirs:
                        continue
                    if not include_hidden and dirname.startswith("."):
                        continue
                    rel_path = rel_dir / dirname
                    if gitignore_spec and is_gitignored(rel_path, cwd, gitignore_spec, is_dir=True):
                        continue
                    filtered_dirs.append(dirname)
                dirnames[:] = filtered_dirs
                if candidate in filenames:
                    if not include_hidden and is_hidden_path(Path(candidate)):
                        continue
                    rel_path = rel_dir / candidate
                    if gitignore_spec and is_gitignored(rel_path, cwd, gitignore_spec, is_dir=False):
                        continue
                    matches.append(Path(root) / candidate)
                if len(matches) > 1:
                    break

            if len(matches) == 1:
                match = matches[0]
                try:
                    return str(match.relative_to(cwd))
                except ValueError:
                    continue

    return None
