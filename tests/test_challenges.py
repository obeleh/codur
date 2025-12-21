"""Pytest harness for Codur challenges."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CHALLENGES_DIR = Path(__file__).resolve().parents[1] / "challenges"


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


def test_challenges_match_expected_output() -> None:
    assert CHALLENGES_DIR.exists(), f"Missing challenges directory: {CHALLENGES_DIR}"
    challenge_dirs = sorted([p for p in CHALLENGES_DIR.iterdir() if p.is_dir()])
    assert challenge_dirs, "No challenges found"

    for challenge_dir in challenge_dirs:
        expected_path = challenge_dir / "expected.txt"
        main_path = challenge_dir / "main.py"
        assert expected_path.exists(), f"Missing expected.txt in {challenge_dir}"
        assert main_path.exists(), f"Missing main.py in {challenge_dir}"

        expected = _read_text(expected_path)
        actual = _run_challenge(main_path)
        assert actual == expected, (
            "Output mismatch in "
            f"{challenge_dir.name}\nExpected:\n{expected}\nActual:\n{actual}"
        )
