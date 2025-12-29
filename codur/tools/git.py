"""
Git tools for Codur.
"""

from __future__ import annotations

from pathlib import Path

import pygit2

from codur.config import CodurConfig
from codur.constants import DEFAULT_MAX_BYTES, DEFAULT_MAX_RESULTS, TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import (
    ToolContext,
    ToolSideEffect,
    tool_contexts,
    tool_scenarios,
    tool_side_effects,
)
from codur.utils.path_utils import resolve_root, resolve_path
from codur.utils.text_helpers import truncate_chars
from codur.utils.validation import require_tool_permission


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


def _resolve_config(config: CodurConfig | None, state: AgentState | None) -> CodurConfig:
    if config is not None:
        return config
    if state is not None and hasattr(state, "get_config"):
        cfg = state.get_config()
        if cfg is not None:
            return cfg
    raise ValueError("Config not available in tool state")


def _require_git_write_enabled(config: CodurConfig) -> None:
    require_tool_permission(
        config,
        "tools.allow_git_write",
        "Git write tools",
        "Git write tools are disabled. Set tools.allow_git_write: true in codur.yaml.",
    )


def _repo_workdir(repo: pygit2.Repository) -> Path:
    if not repo.workdir:
        raise ValueError("Bare repositories are not supported")
    return Path(repo.workdir).resolve()


def _resolve_repo_path(
    raw_path: str,
    repo_root: Path,
    root: str | Path | None,
    allow_outside_root: bool,
) -> tuple[Path, Path]:
    base_root = resolve_root(root) if root else repo_root
    abs_path = resolve_path(raw_path, base_root, allow_outside_root=allow_outside_root)
    try:
        rel_path = abs_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path is outside repository: {raw_path}") from exc
    return abs_path, rel_path


def _resolve_signature(
    repo: pygit2.Repository,
    name: str | None,
    email: str | None,
    role: str,
) -> pygit2.Signature:
    if name and email:
        return pygit2.Signature(name, email)
    try:
        return repo.default_signature
    except Exception as exc:
        raise ValueError(
            f"{role} name/email not set. Provide {role}_name/{role}_email or set git user.name/user.email."
        ) from exc


@tool_scenarios(TaskType.EXPLANATION, TaskType.CODE_FIX, TaskType.REFACTOR)
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
        detached = repo.head_is_detached
        branch = head.shorthand if not detached else "HEAD"
        if not repo.head_is_unborn:
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


@tool_scenarios(TaskType.EXPLANATION, TaskType.CODE_FIX, TaskType.REFACTOR)
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
        if repo.head is None or repo.head_is_unborn:
            raise ValueError("HEAD is not available for staged diff")
        diff = repo.diff("HEAD", cached=True, context_lines=context_lines, interhunk_lines=interhunk_lines)
    elif normalized_mode == "all":
        if repo.head is None or repo.head_is_unborn:
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
    return truncate_chars(diff_text, max_chars=max_bytes)


@tool_scenarios(TaskType.EXPLANATION, TaskType.REFACTOR)
def git_log(
    max_count: int = 20,
    root: str | Path | None = None,
    state: AgentState | None = None,
) -> list[dict]:
    """
    Return recent commit metadata for the current repository.
    """
    repo = _open_repo(root)
    if repo.head is None or repo.head_is_unborn:
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


@tool_side_effects(ToolSideEffect.STATE_CHANGE)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def git_stage_files(
    paths: list[str],
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> dict:
    """
    Stage file paths for commit (git add).
    """
    if not paths:
        raise ValueError("git_stage_files requires a non-empty 'paths' list")

    config = _resolve_config(config, state)
    _require_git_write_enabled(config)

    repo = _open_repo(root)
    repo_root = _repo_workdir(repo)
    index = repo.index

    added: list[str] = []
    removed: list[str] = []
    staged_dirs: list[str] = []
    errors: list[str] = []

    for path in paths:
        try:
            abs_path, rel_path = _resolve_repo_path(path, repo_root, root, allow_outside_root)
            rel_str = str(rel_path)
            if abs_path.is_dir():
                index.add_all([rel_str])
                staged_dirs.append(rel_str)
            elif abs_path.exists():
                index.add(rel_str)
                added.append(rel_str)
            else:
                index.remove(rel_str)
                removed.append(rel_str)
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    index.write()
    return {
        "added": added,
        "removed": removed,
        "staged_dirs": staged_dirs,
        "errors": errors,
    }


@tool_side_effects(ToolSideEffect.STATE_CHANGE)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def git_stage_all(
    root: str | Path | None = None,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> dict:
    """
    Stage all changes (git add -A).
    """
    config = _resolve_config(config, state)
    _require_git_write_enabled(config)

    repo = _open_repo(root)
    index = repo.index
    index.add_all()
    index.write()
    return {"staged": True}


@tool_side_effects(ToolSideEffect.STATE_CHANGE)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.CODE_FIX, TaskType.REFACTOR)
def git_commit(
    message: str,
    root: str | Path | None = None,
    allow_empty: bool = False,
    author_name: str | None = None,
    author_email: str | None = None,
    committer_name: str | None = None,
    committer_email: str | None = None,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> dict:
    """
    Create a git commit from the current index.
    """
    if not message:
        raise ValueError("git_commit requires a commit message")

    config = _resolve_config(config, state)
    _require_git_write_enabled(config)

    repo = _open_repo(root)
    repo_root = _repo_workdir(repo)
    index = repo.index
    index.write()
    tree_id = index.write_tree()

    if repo.head_is_unborn:
        if not allow_empty and len(index) == 0:
            raise ValueError("Cannot create empty initial commit without allow_empty=true")
        parents: list[pygit2.Oid] = []
        ref_name = repo.head.name if repo.head is not None else "HEAD"
    else:
        if not allow_empty:
            diff = repo.diff("HEAD", cached=True)
            if diff.stats.files_changed == 0:
                raise ValueError("No staged changes to commit (use allow_empty=true to override)")
        parents = [repo.head.target]
        ref_name = repo.head.name

    author = _resolve_signature(repo, author_name, author_email, "author")
    committer = _resolve_signature(repo, committer_name, committer_email, "committer")

    commit_id = repo.create_commit(ref_name, author, committer, message, tree_id, parents)
    return {
        "commit": str(commit_id),
        "repo_root": str(repo_root),
        "ref": ref_name,
    }
