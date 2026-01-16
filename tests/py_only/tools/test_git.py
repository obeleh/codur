import pytest
import pygit2
from pathlib import Path
from codur.tools.git import (
    git_status, git_stage_files, git_stage_all, git_commit, git_log, git_diff
)
from dataclasses import dataclass, field

@dataclass
class MockToolsConfig:
    allow_git_write: bool = True

@dataclass
class MockConfig:
    tools: MockToolsConfig = field(default_factory=MockToolsConfig)

@dataclass
class MockToolsConfigNoWrite:
    allow_git_write: bool = False

@dataclass
class MockConfigNoWrite:
    tools: MockToolsConfigNoWrite = field(default_factory=MockToolsConfigNoWrite)

@pytest.fixture
def mock_config():
    return MockConfig()

@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    
    # Init repo
    repo = pygit2.init_repository(str(repo_path))
    
    # Configure user
    repo.config["user.name"] = "Test User"
    repo.config["user.email"] = "test@example.com"
    
    # Create a file
    (repo_path / "file1.txt").write_text("Initial content", encoding="utf-8")
    
    # Commit it
    index = repo.index
    index.add("file1.txt")
    index.write()
    tree = index.write_tree()
    author = pygit2.Signature("Test User", "test@example.com")
    repo.create_commit("HEAD", author, author, "Initial commit", tree, [])
    
    return repo_path

def test_git_status(git_repo):
    # Create a new file
    (git_repo / "new.txt").write_text("New file", encoding="utf-8")
    
    status = git_status(root=git_repo)
    assert "new.txt" in status["untracked"]
    assert status["counts"]["untracked"] == 1

def test_git_stage_files(git_repo, mock_config):
    (git_repo / "new.txt").write_text("New file", encoding="utf-8")
    
    result = git_stage_files(["new.txt"], root=git_repo, config=mock_config)
    assert "new.txt" in result["added"]
    
    status = git_status(root=git_repo)
    assert "new.txt" in status["staged"]

def test_git_commit(git_repo, mock_config):
    (git_repo / "new.txt").write_text("New file", encoding="utf-8")
    git_stage_files(["new.txt"], root=git_repo, config=mock_config)
    
    result = git_commit("Add new file", root=git_repo, config=mock_config)
    assert result["commit"] is not None
    
    log = git_log(root=git_repo)
    assert log[0]["summary"] == "Add new file"

def test_git_log(git_repo):
    log = git_log(root=git_repo)
    assert len(log) == 1
    assert log[0]["summary"] == "Initial commit"
    assert log[0]["author"]["name"] == "Test User"

def test_git_diff(git_repo):
    # Modify a file
    (git_repo / "file1.txt").write_text("Modified content", encoding="utf-8")
    
    diff = git_diff(root=git_repo, mode="unstaged")
    assert "Initial content" in diff
    assert "Modified content" in diff


def test_git_stage_all(git_repo, mock_config):
    (git_repo / "new.txt").write_text("New file", encoding="utf-8")
    result = git_stage_all(root=git_repo, config=mock_config)
    assert result["staged"] is True
    status = git_status(root=git_repo)
    assert "new.txt" in status["staged"]


def test_git_stage_all_requires_write(git_repo):
    (git_repo / "new.txt").write_text("New file", encoding="utf-8")
    with pytest.raises(ValueError, match="Git write tools are disabled"):
        git_stage_all(root=git_repo, config=MockConfigNoWrite())


def test_git_stage_files_rejects_outside_root(tmp_path, mock_config):
    """Test that git_stage_files reports errors for paths outside the workspace root (when allow_outside_root=False)."""
    root = tmp_path / "repo"
    root.mkdir()
    repo = pygit2.init_repository(str(root))
    repo.config["user.name"] = "Test User"
    repo.config["user.email"] = "test@example.com"

    outside = tmp_path / "outside.txt"
    outside.write_text("content", encoding="utf-8")

    # Note: git_stage_files collects errors instead of raising
    result = git_stage_files([str(outside)], root=root, config=mock_config, allow_outside_root=False)
    assert len(result["errors"]) > 0
    assert any("Path escapes workspace root" in err or "Path is outside repository" in err for err in result["errors"])


def test_git_stage_files_relative_path(git_repo, mock_config):
    """Test that git_stage_files works with relative paths."""
    (git_repo / "new.txt").write_text("New file", encoding="utf-8")

    result = git_stage_files(["new.txt"], root=git_repo, config=mock_config)
    assert "new.txt" in result["added"]


def test_git_stage_files_dotdot_path(tmp_path, mock_config):
    """Test that ../ in path is validated correctly."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "sub").mkdir()
    repo = pygit2.init_repository(str(root))
    repo.config["user.name"] = "Test User"
    repo.config["user.email"] = "test@example.com"

    # Note: git_stage_files collects errors instead of raising
    result = git_stage_files(["../outside.txt"], root=root / "sub", config=mock_config, allow_outside_root=False)
    assert len(result["errors"]) > 0
    assert any("Path escapes workspace root" in err or "Path is outside repository" in err for err in result["errors"])
