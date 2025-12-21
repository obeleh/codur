"""Pytest harness for Codur challenges."""

from __future__ import annotations

import os
import warnings
import subprocess
import sys
from pathlib import Path

import pygit2


CHALLENGES_DIR = Path(__file__).resolve().parents[1] / "challenges"
REPO_ROOT = CHALLENGES_DIR.parent

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _run_challenge(main_path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(main_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise AssertionError(f"Challenge failed: {main_path}\nstdout: {output}\nstderr: {stderr}")
    return output


def _run_codur(prompt: str, cwd: Path) -> str:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "codur.cli",
            "--command",
            prompt,
            "--raw",
            "--config",
            str(REPO_ROOT / "codur.yaml"),
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    stdout = (result.stdout or "").strip()
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise AssertionError(f"codur failed in {cwd}\nstdout: {stdout}\nstderr: {stderr}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    filtered = [line for line in lines if not line.lower().startswith("selected agent:")]
    return "\n".join(filtered).strip()


def _reset_challenges() -> None:
    repo_path = pygit2.discover_repository(str(REPO_ROOT))
    if repo_path is None:
        raise AssertionError("Could not locate git repository for challenges reset")
    repo = pygit2.Repository(repo_path)
    head = repo.revparse_single("HEAD")
    challenge_prefix = f"{CHALLENGES_DIR.name}/"

    for path, status in repo.status().items():
        if not path.startswith(challenge_prefix):
            continue
        if path == f"{challenge_prefix}README.md":
            continue
        full_path = REPO_ROOT / path
        if status & pygit2.GIT_STATUS_WT_NEW:
            if full_path.is_dir():
                for child in sorted(full_path.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                for child in sorted(full_path.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                full_path.rmdir()
            elif full_path.exists():
                full_path.unlink()
            continue
        repo.checkout_tree(head, paths=[path], strategy=pygit2.GIT_CHECKOUT_FORCE)


def test_challenges_match_expected_output() -> None:
    assert CHALLENGES_DIR.exists(), f"Missing challenges directory: {CHALLENGES_DIR}"
    challenge_dirs = sorted([p for p in CHALLENGES_DIR.iterdir() if p.is_dir()])
    assert challenge_dirs, "No challenges found"

    try:
        for challenge_dir in challenge_dirs:
            expected_path = challenge_dir / "expected.txt"
            main_path = challenge_dir / "main.py"
            prompt_path = challenge_dir / "prompt.txt"
            assert expected_path.exists(), f"Missing expected.txt in {challenge_dir}"
            assert main_path.exists(), f"Missing main.py in {challenge_dir}"
            assert prompt_path.exists(), f"Missing prompt.txt in {challenge_dir}"
            allowed = {"expected.txt", "main.py", "prompt.txt"}
            extras = [p.name for p in challenge_dir.iterdir() if p.is_file() and p.name not in allowed]
            assert not extras, f"Unexpected files in {challenge_dir}: {extras}"

            prompt = _read_text(prompt_path)
            _run_codur(prompt, cwd=challenge_dir)

            expected = _read_text(expected_path)
            actual = _run_challenge(main_path)
            assert actual == expected, (
                "Output mismatch in "
                f"{challenge_dir.name}\nExpected:\n{expected}\nActual:\n{actual}"
            )
    finally:
        _reset_challenges()
