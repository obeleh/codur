"""Markdown-specific tools for parsing and manipulating markdown files.

This module provides tools for analyzing and extracting structured data from
markdown documents, including headers, sections, and tables.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

from codur.graph.state import AgentState
from codur.utils.path_utils import resolve_path, resolve_root
from codur.utils.ignore_utils import get_config_from_state
from codur.utils.validation import validate_file_access


def markdown_outline(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> str:
    """Extract headers from a markdown file to show document structure.

    Args:
        path: Path to markdown file

    Returns:
        Formatted outline showing headers with indentation based on level

    Example:
        >>> content = "# Title\\n## Section 1\\n### Subsection\\n## Section 2"
        >>> # Returns:
        >>> # # Title
        >>> #   ## Section 1
        >>> #     ### Subsection
        >>> #   ## Section 2
    """
    raw_path = Path(path)
    effective_allow_outside_root = allow_outside_root or (root is None and raw_path.is_absolute())
    if root is None and raw_path.is_absolute():
        target = raw_path.resolve()
    else:
        target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config_from_state(state),
        operation="read",
        allow_outside_root=effective_allow_outside_root,
    )
    content = target.read_text(encoding="utf-8", errors="replace")
    lines = content.split('\n')

    outline_lines = []
    for line in lines:
        # Match markdown headers: # Header, ## Header, etc.
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            indent = '  ' * (level - 1)
            outline_lines.append(f"{indent}{'#' * level} {title}")

    if not outline_lines:
        return "No headers found in markdown file"

    return '\n'.join(outline_lines)


def markdown_extract_sections(
    path: str,
    section_names: List[str],
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> Dict[str, str]:
    """Extract specific sections from a markdown file by header name.

    Extracts the content under each specified header, including any subsections,
    until the next header of the same or higher level is encountered.

    Args:
        path: Path to markdown file
        section_names: List of section header names to extract (without # symbols)

    Returns:
        Dictionary mapping section names to their full content (including the header)

    Example:
        >>> extract_sections("README.md", ["Installation", "Usage"])
        >>> # Returns: {"Installation": "## Installation\\n...\\n", "Usage": "## Usage\\n...\\n"}
    """
    raw_path = Path(path)
    effective_allow_outside_root = allow_outside_root or (root is None and raw_path.is_absolute())
    if root is None and raw_path.is_absolute():
        target = raw_path.resolve()
    else:
        target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config_from_state(state),
        operation="read",
        allow_outside_root=effective_allow_outside_root,
    )
    content = target.read_text(encoding="utf-8", errors="replace")
    lines = content.split('\n')

    sections = {}
    current_section = None
    current_content = []
    current_level = None

    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())

        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Save previous section if it was one we're tracking
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()

            # Check if this is a section we want
            if title in section_names:
                current_section = title
                current_level = level
                current_content = [line]
            elif current_section and level <= current_level:
                # We've moved to a new top-level section, stop tracking
                current_section = None
                current_content = []
                current_level = None
            elif current_section:
                # Continue collecting content for current section
                current_content.append(line)
        elif current_section:
            current_content.append(line)

    # Save final section if we're still tracking one
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections


def markdown_extract_tables(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> List[Dict[str, Any]]:
    """Extract markdown tables from a file and convert to structured data.

    Parses markdown pipe tables into a structured format with headers and rows.

    Args:
        path: Path to markdown file

    Returns:
        List of tables, each as {"headers": [...], "rows": [[...]]}

    Example:
        >>> # For a table like:
        >>> # | Name | Age |
        >>> # | ---- | --- |
        >>> # | Alice | 30 |
        >>> # | Bob   | 25 |
        >>> # Returns: [{"headers": ["Name", "Age"], "rows": [["Alice", "30"], ["Bob", "25"]]}]
    """
    raw_path = Path(path)
    effective_allow_outside_root = allow_outside_root or (root is None and raw_path.is_absolute())
    if root is None and raw_path.is_absolute():
        target = raw_path.resolve()
    else:
        target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        target,
        resolve_root(root),
        get_config_from_state(state),
        operation="read",
        allow_outside_root=effective_allow_outside_root,
    )
    content = target.read_text(encoding="utf-8", errors="replace")
    lines = content.split('\n')

    tables = []
    in_table = False
    current_table = None

    for line in lines:
        # Check if line looks like a table row: | col1 | col2 |
        if '|' in line and line.strip():
            cells = [cell.strip() for cell in line.split('|')[1:-1]]

            if not in_table:
                # First row is headers
                in_table = True
                current_table = {"headers": cells, "rows": []}
            elif re.match(r'^[\s|:-]+$', line):
                # Separator row (| --- | --- |), skip it
                continue
            else:
                # Data row
                current_table["rows"].append(cells)
        elif in_table:
            # End of table (empty line or non-table content)
            tables.append(current_table)
            in_table = False
            current_table = None

    # Save final table if file ends with one
    if in_table and current_table:
        tables.append(current_table)

    return tables


# Export all tools for tool registry
__all__ = [
    "markdown_outline",
    "markdown_extract_sections",
    "markdown_extract_tables",
]
