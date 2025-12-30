"""Text processing and truncation utilities."""

from __future__ import annotations


def truncate_lines(text: str, max_lines: int = 50) -> str:
    """Truncate text to maximum number of lines.

    Args:
        text: Text to truncate
        max_lines: Maximum number of lines to keep

    Returns:
        str: Truncated text with ellipsis if truncated
    """
    lines = text.split('\n')
    if len(lines) <= max_lines:
        return text

    truncated = '\n'.join(lines[:max_lines])
    return f"{truncated}\n... ({len(lines) - max_lines} more lines)"


def truncate_chars(text: str, max_chars: int = 1000) -> str:
    """Truncate text to maximum number of characters.

    Args:
        text: Text to truncate
        max_chars: Maximum number of characters to keep

    Returns:
        str: Truncated text with ellipsis if truncated
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    return f"{truncated}... ({len(text) - max_chars} more chars)"


def smart_truncate(text: str, target_length: int = 1000) -> str:
    """Truncate text intelligently, preserving line structure.

    Tries to end at a complete sentence or paragraph boundary.

    Args:
        text: Text to truncate
        target_length: Target character count (approximate)

    Returns:
        str: Truncated text
    """
    if len(text) <= target_length:
        return text

    # First, truncate to target length
    truncated = text[:target_length]

    # Try to find a sentence boundary (period, question mark, exclamation)
    for terminator in ['. ', '? ', '! ', '\n\n']:
        last_idx = truncated.rfind(terminator)
        if last_idx > target_length * 0.8:  # At least 80% of target
            return truncated[:last_idx + len(terminator) - 1] + "\n..."

    # Fall back to truncating at last newline
    last_newline = truncated.rfind('\n')
    if last_newline > target_length * 0.7:  # At least 70% of target
        return truncated[:last_newline] + "\n..."

    # No good boundary found, just truncate and add ellipsis
    return truncated + "..."
