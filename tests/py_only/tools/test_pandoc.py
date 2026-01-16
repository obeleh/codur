"""Tests for pandoc tool wrapper."""

import shutil
import subprocess
from dataclasses import dataclass

import pytest

from codur.tools.pandoc import convert_document


def test_convert_document_requires_pandoc(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    source = tmp_path / "doc.md"
    source.write_text("# Title\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="pandoc is not available"):
        convert_document(str(source), "pdf", root=tmp_path)


def test_convert_document_success_and_failure(tmp_path, monkeypatch):
    @dataclass
    class FakeResult:
        returncode: int
        stdout: str = ""
        stderr: str = ""

    source = tmp_path / "doc.md"
    source.write_text("# Title\n", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pandoc")

    def fake_run_success(cmd, capture_output, text, check):
        return FakeResult(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run_success)
    result = convert_document(str(source), "pdf", root=tmp_path)
    assert result["output"].endswith(".pdf")
    assert "pandoc" in result["command"]

    def fake_run_fail(cmd, capture_output, text, check):
        return FakeResult(returncode=1, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run_fail)
    with pytest.raises(RuntimeError, match="pandoc failed: boom"):
        convert_document(str(source), "pdf", root=tmp_path)


def test_convert_document_rejects_outside_root(tmp_path, monkeypatch):
    """Test that convert_document rejects input paths outside the workspace root."""
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Title", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pandoc")

    with pytest.raises(ValueError, match="Path escapes workspace root"):
        convert_document(str(outside), "pdf", root=root)


def test_convert_document_output_rejects_outside_root(tmp_path, monkeypatch):
    """Test that convert_document rejects output paths outside the workspace root."""
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "doc.md"
    source.write_text("# Title", encoding="utf-8")
    outside_output = tmp_path / "outside.pdf"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pandoc")

    with pytest.raises(ValueError, match="Path escapes workspace root"):
        convert_document(str(source), "pdf", output_path=str(outside_output), root=root)


def test_convert_document_relative_path(tmp_path, monkeypatch):
    """Test that convert_document works with relative paths."""
    @dataclass
    class FakeResult:
        returncode: int
        stdout: str = ""
        stderr: str = ""

    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "doc.md"
    source.write_text("# Title", encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pandoc")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeResult(returncode=0))

    result = convert_document("doc.md", "pdf", root=root)
    assert result["output"].endswith(".pdf")


def test_convert_document_dotdot_path(tmp_path, monkeypatch):
    """Test that ../ in path is validated correctly."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sub").mkdir()

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/pandoc")

    with pytest.raises(ValueError, match="Path escapes workspace root"):
        convert_document("../outside.md", "pdf", root=root / "sub")
