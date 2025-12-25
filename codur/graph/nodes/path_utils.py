"""Consolidated path extraction and validation utilities for graph nodes."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from codur.tools.filesystem import EXCLUDE_DIRS


def looks_like_path(token: str) -> bool:
    """Check if a token appears to be a file path.

    Args:
        token: Token to check

    Returns:
        bool: True if token looks like a path
    """
    if token.startswith("@"):
        return True
    if "/" in token or "\\" in token:
        return True
    if re.search(r"\.[A-Za-z0-9]{1,5}$", token):
        return True
    return False


def extract_path_from_message(text: str) -> Optional[str]:
    """Extract first file path from message text.

    Tries multiple patterns:
    1. @filepath syntax
    2. "in filepath" or "inside filepath" syntax
    3. .py file reference

    Args:
        text: Text to search

    Returns:
        Optional[str]: First path found, or None
    """
    at_match = re.search(r"@([^\s,]+)", text)
    if at_match:
        return at_match.group(1)

    in_match = re.search(r"(?:in|inside)\s+([^\s,]+)", text, re.IGNORECASE)
    if in_match:
        candidate = in_match.group(1).strip().strip(".,:;()[]{}")
        if looks_like_path(candidate):
            return candidate.lstrip("@")
        return None

    path_match = re.search(r"([^\s,]+\.py)", text)
    if path_match:
        return path_match.group(1)

    return None


def extract_file_paths(text: str) -> list[str]:
    """Extract all file paths from text.

    Recognizes multiple path patterns:
    - @filepath syntax
    - Files with common extensions (py, js, ts, json, yaml, etc.)
    - Quoted paths

    Args:
        text: Text to search

    Returns:
        list[str]: List of unique paths found (duplicates removed)
    """
    paths = []

    # @file.py syntax
    at_matches = re.findall(r"@([^\s,]+)", text)
    paths.extend(at_matches)

    # Explicit file extensions
    ext_matches = re.findall(r"([^\s,\"']+\.(?:json|yaml|html|css|yml|txt|py|js|ts|md))", text)
    paths.extend(ext_matches)

    # Quoted paths
    quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
    for q in quoted:
        if "/" in q or "." in q:
            paths.append(q)

    # Clean paths: strip leading '@' and remove duplicates
    cleaned_paths = {p.lstrip('@') for p in paths}

    return list(cleaned_paths)


def find_workspace_match(raw_text: str) -> Optional[str]:
    """Find a file in workspace that matches references in text.

    Intelligently resolves file references against the workspace:
    1. Checks if path is absolute and within workspace
    2. Resolves relative paths
    3. Searches workspace for matching filename

    Args:
        raw_text: Text that may contain file references

    Returns:
        Optional[str]: Relative path to matching file, or None
    """
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
                dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
                if candidate in filenames:
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
