import os
from pathlib import Path

import pytest

from codur.tools.psutil_tools import (
    system_cpu_stats,
    system_memory_stats,
    system_disk_usage,
    system_process_snapshot,
    system_processes_top,
    system_processes_list,
)


def test_system_cpu_stats():
    result = system_cpu_stats()
    assert "percent" in result
    assert "counts" in result
    assert result["counts"]["logical"] >= 0
    assert result["counts"]["physical"] >= 0


def test_system_memory_stats():
    result = system_memory_stats()
    assert result["virtual"]["total"] > 0
    assert 0.0 <= result["virtual"]["percent"] <= 100.0


def test_system_disk_usage(tmp_path):
    result = system_disk_usage(path=".", root=tmp_path)
    assert Path(result["path"]).exists()
    assert result["total"] > 0
    assert 0.0 <= result["percent"] <= 100.0


def test_system_process_snapshot_current():
    result = system_process_snapshot()
    assert result["pid"] == os.getpid()
    assert result["name"]


def test_system_process_snapshot_invalid_pid():
    with pytest.raises(ValueError):
        system_process_snapshot(pid=999999)


def test_system_processes_top_limit():
    result = system_processes_top(limit=3)
    assert len(result) <= 3
    for entry in result:
        assert "pid" in entry
        assert "name" in entry


def test_system_processes_list_limit():
    result = system_processes_list(limit=5)
    assert len(result) <= 5
    for entry in result:
        assert "pid" in entry
        assert "name" in entry
        assert "status" in entry


def test_system_processes_list_detailed():
    result = system_processes_list(limit=3, detailed=True)
    assert len(result) <= 3
    for entry in result:
        assert "pid" in entry
        assert "name" in entry
        assert "status" in entry
        assert "cpu_percent" in entry
        assert "memory_percent" in entry
        assert "create_time" in entry
        assert "username" in entry
        assert "cmdline" in entry
