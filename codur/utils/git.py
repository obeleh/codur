"""Git utilities."""

from __future__ import annotations

from pathlib import Path

import pygit2


def _colorize_unified_diff(diff_text: str) -> str:
    lines = []
    for line in diff_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            lines.append(f"\033[1;37m{line}\033[0m")
        elif line.startswith("@@ "):
            lines.append(f"\033[1;36m{line}\033[0m")
        elif line.startswith("+"):
            lines.append(f"\033[32m{line}\033[0m")
        elif line.startswith("-"):
            lines.append(f"\033[31m{line}\033[0m")
        else:
            lines.append(line)
    return "\n".join(lines)


def get_diff_for_path(repo_root: Path, rel_path: Path | str, *, colorize: bool = False) -> str:
    """Return a unified diff for a path relative to repo_root.

    Uses pygit2 to avoid shelling out to git. Colorization can be layered
    on top by callers if desired.
    """
    repo_path = pygit2.discover_repository(str(repo_root))
    if repo_path is None:
        return ""
    repo = pygit2.Repository(repo_path)
    head = repo.revparse_single("HEAD")
    diff = repo.diff(head.tree, None)
    output: list[str] = []
    for patch in diff:
        delta = patch.delta
        new_path = delta.new_file.path if delta.new_file else ""
        old_path = delta.old_file.path if delta.old_file else ""
        if str(rel_path) not in (new_path, old_path):
            continue
        text = getattr(patch, "text", None) or str(patch)
        text = text.rstrip()
        if text:
            output.append(text)
    diff_text = "\n".join(output)
    if colorize and diff_text:
        return _colorize_unified_diff(diff_text)
    return diff_text
