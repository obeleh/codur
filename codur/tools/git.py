"""
Git tools for Codur.
"""

from __future__ import annotations

from pathlib import Path

import pygit2

from codur.constants import DEFAULT_MAX_BYTES, DEFAULT_MAX_RESULTS
from codur.graph.state import AgentState
from codur.utils.path_utils import resolve_root


_INDEX_FLAGS = (
    pygit2.GIT_STATUS_INDEX_NEW
    | pygit2.GIT_STATUS_INDEX_MODIFIED
    | pygit2.GIT_STATUS_INDEX_DELETED
    | pygit2.GIT_STATUS_INDEX_RENAMED
    | pygit2.GIT_STATUS_INDEX_TYPECHANGE
)
_WT_FLAGS = (
    pygit2.GIT_STATUS_WT_NEW
    | pygit2.GIT_STATUS_WT_MODIFIED
    | pygit2.GIT_STATUS_WT_DELETED
    | pygit2.GIT_STATUS_WT_RENAMED
    | pygit2.GIT_STATUS_WT_TYPECHANGE
    | pygit2.GIT_STATUS_WT_UNREADABLE
)
_WT_NON_NEW = _WT_FLAGS & ~pygit2.GIT_STATUS_WT_NEW


def _open_repo(root: str | Path | None) -> pygit2.Repository:
    root_path = resolve_root(root)
    repo_path = pygit2.discover_repository(str(root_path))
    if repo_path is None:
        raise ValueError("No git repository found under root")
    return pygit2.Repository(repo_path)


def _truncate(text: str, max_bytes: int) -> str:
    if len(text) <= max_bytes:
        return text
    return text[:max_bytes] + "\n... [truncated]"


def git_status(
    root: str | Path | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    state: AgentState | None = None,
) -> dict:
    """
    Return a summarized git status for the current repository.
    """
    repo = _open_repo(root)
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    conflicted: list[str] = []

    for path, flags in repo.status().items():
        if flags & pygit2.GIT_STATUS_CONFLICTED:
            conflicted.append(path)
        if flags & _INDEX_FLAGS:
            staged.append(path)
        if flags & _WT_FLAGS:
            if flags & pygit2.GIT_STATUS_WT_NEW and not (flags & _INDEX_FLAGS):
                untracked.append(path)
            elif flags & _WT_NON_NEW:
                unstaged.append(path)

    staged.sort()
    unstaged.sort()
    untracked.sort()
    conflicted.sort()

    try:
        head = repo.head
    except Exception:
        head = None
    branch = ""
    head_sha = ""
    detached = False
    if head is not None:
        detached = head.is_detached
        branch = head.shorthand if not detached else "HEAD"
        if not head.is_unborn:
            head_sha = str(head.target)

    def _clip(items: list[str]) -> list[str]:
        return items[:max_results] if max_results > 0 else items

    return {
        "repo_root": str(Path(repo.workdir).resolve()) if repo.workdir else "",
        "branch": branch,
        "head": head_sha,
        "detached": detached,
        "counts": {
            "staged": len(staged),
            "unstaged": len(unstaged),
            "untracked": len(untracked),
            "conflicted": len(conflicted),
        },
        "staged": _clip(staged),
        "unstaged": _clip(unstaged),
        "untracked": _clip(untracked),
        "conflicted": _clip(conflicted),
    }


def git_diff(
    path: str | None = None,
    mode: str = "unstaged",
    root: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    context_lines: int = 3,
    interhunk_lines: int = 0,
    state: AgentState | None = None,
) -> str:
    """
    Return a unified diff for the current repo (unstaged, staged, or all).
    """
    repo = _open_repo(root)
    normalized_mode = mode.lower().strip()
    if normalized_mode == "unstaged":
        diff = repo.diff(context_lines=context_lines, interhunk_lines=interhunk_lines)
    elif normalized_mode == "staged":
        if repo.head is None or repo.head.is_unborn:
            raise ValueError("HEAD is not available for staged diff")
        diff = repo.diff("HEAD", cached=True, context_lines=context_lines, interhunk_lines=interhunk_lines)
    elif normalized_mode == "all":
        if repo.head is None or repo.head.is_unborn:
            raise ValueError("HEAD is not available for diff against HEAD")
        diff = repo.diff("HEAD", context_lines=context_lines, interhunk_lines=interhunk_lines)
    else:
        raise ValueError("mode must be one of: unstaged, staged, all")

    chunks: list[str] = []
    for patch in diff:
        delta = patch.delta
        new_path = delta.new_file.path if delta.new_file else ""
        old_path = delta.old_file.path if delta.old_file else ""
        if path and path not in (new_path, old_path):
            continue
        text = getattr(patch, "text", None) or str(patch)
        text = text.rstrip()
        if text:
            chunks.append(text)

    diff_text = "\n".join(chunks)
    return _truncate(diff_text, max_bytes)


def git_log(
    max_count: int = 20,
    root: str | Path | None = None,
    state: AgentState | None = None,
) -> list[dict]:
    """
    Return recent commit metadata for the current repository.
    """
    repo = _open_repo(root)
    if repo.head is None or repo.head.is_unborn:
        return []

    commits: list[dict] = []
    for commit in repo.walk(repo.head.target, pygit2.GIT_SORT_TIME):
        commits.append({
            "id": str(commit.id),
            "summary": (commit.message or "").splitlines()[0] if commit.message else "",
            "author": {
                "name": commit.author.name,
                "email": commit.author.email,
                "time": commit.author.time,
            },
            "committer": {
                "name": commit.committer.name,
                "email": commit.committer.email,
                "time": commit.committer.time,
            },
        })
        if len(commits) >= max_count:
            break
    return commits
