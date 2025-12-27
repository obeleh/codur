"""System inspection tools powered by psutil."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psutil

from codur.constants import TaskType
from codur.graph.state import AgentState
from codur.tools.tool_annotations import tool_scenarios
from codur.utils.path_utils import resolve_path


@tool_scenarios(TaskType.CODE_VALIDATION, TaskType.EXPLANATION)
def system_cpu_stats(
    per_cpu: bool = False,
    interval: float = 0.0,
    state: AgentState | None = None,
) -> dict[str, Any]:
    """Return CPU usage summary and core counts."""
    interval = max(0.0, float(interval))
    try:
        percent = psutil.cpu_percent(interval=interval, percpu=per_cpu)
    except (psutil.Error, OSError, PermissionError, SystemError):
        percent = 0.0 if not per_cpu else []

    logical = None
    physical = None
    try:
        logical = psutil.cpu_count(logical=True)
    except (psutil.Error, OSError, PermissionError, SystemError):
        logical = os.cpu_count()
    try:
        physical = psutil.cpu_count(logical=False)
    except (psutil.Error, OSError, PermissionError, SystemError):
        physical = None

    counts = {
        "logical": int(logical or 0),
        "physical": int(physical or 0),
    }
    load_avg = None
    if hasattr(os, "getloadavg"):
        try:
            load_avg = os.getloadavg()
        except OSError:
            load_avg = None
    return {
        "percent": percent,
        "counts": counts,
        "load_avg": load_avg,
    }


@tool_scenarios(TaskType.CODE_VALIDATION, TaskType.EXPLANATION)
def system_memory_stats(state: AgentState | None = None) -> dict[str, Any]:
    """Return virtual and swap memory usage."""
    try:
        virtual = psutil.virtual_memory()
    except (psutil.Error, OSError, PermissionError, SystemError):
        virtual = None
    try:
        swap = psutil.swap_memory()
    except (psutil.Error, OSError, PermissionError, SystemError):
        swap = None
    return {
        "virtual": {
            "total": int(getattr(virtual, "total", 0)),
            "available": int(getattr(virtual, "available", 0)),
            "used": int(getattr(virtual, "used", 0)),
            "percent": float(getattr(virtual, "percent", 0.0)),
        },
        "swap": {
            "total": int(getattr(swap, "total", 0)),
            "used": int(getattr(swap, "used", 0)),
            "free": int(getattr(swap, "free", 0)),
            "percent": float(getattr(swap, "percent", 0.0)),
        },
    }


@tool_scenarios(TaskType.CODE_VALIDATION, TaskType.FILE_OPERATION)
def system_disk_usage(
    path: str = ".",
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict[str, Any]:
    """Return disk usage for a path."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    usage = psutil.disk_usage(str(target))
    return {
        "path": str(target),
        "total": int(usage.total),
        "used": int(usage.used),
        "free": int(usage.free),
        "percent": float(usage.percent),
    }


@tool_scenarios(TaskType.CODE_VALIDATION)
def system_process_snapshot(
    pid: int | None = None,
    state: AgentState | None = None,
) -> dict[str, Any]:
    """Return a summary for a single process."""
    target_pid = pid if pid is not None else os.getpid()
    try:
        process = psutil.Process(int(target_pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        raise ValueError(f"Process not available: {target_pid}") from exc
    try:
        memory_info = process.memory_info()
    except (psutil.Error, OSError, PermissionError, SystemError):
        memory_info = None
    try:
        cpu_percent = float(process.cpu_percent(interval=0.0))
    except (psutil.Error, OSError, PermissionError, SystemError):
        cpu_percent = 0.0
    try:
        memory_percent = float(process.memory_percent())
    except (psutil.Error, OSError, PermissionError, SystemError):
        memory_percent = 0.0
    return {
        "pid": process.pid,
        "name": process.name(),
        "status": process.status(),
        "cpu_percent": cpu_percent,
        "memory_rss": int(getattr(memory_info, "rss", 0)),
        "memory_percent": memory_percent,
        "create_time": float(process.create_time()),
    }


@tool_scenarios(TaskType.CODE_VALIDATION)
def system_processes_top(
    limit: int = 5,
    sort_by: str = "cpu",
    state: AgentState | None = None,
) -> list[dict[str, Any]]:
    """Return a small list of top processes by CPU or memory."""
    limit = max(1, int(limit))
    sort_key = sort_by.lower()
    if sort_key not in ("cpu", "memory", "pid"):
        raise ValueError("sort_by must be one of: cpu, memory, pid")

    entries = []
    try:
        iterator = psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"])
    except (psutil.Error, OSError, PermissionError, SystemError):
        return []
    try:
        for process in iterator:
            info = process.info
            entries.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name") or "",
                    "cpu_percent": float(info.get("cpu_percent") or 0.0),
                    "memory_percent": float(info.get("memory_percent") or 0.0),
                }
            )
    except (psutil.Error, OSError, PermissionError, SystemError):
        return []

    if sort_key == "cpu":
        entries.sort(key=lambda item: item["cpu_percent"], reverse=True)
    elif sort_key == "memory":
        entries.sort(key=lambda item: item["memory_percent"], reverse=True)
    else:
        entries.sort(key=lambda item: item["pid"] or 0)

    return entries[:limit]


@tool_scenarios(TaskType.CODE_VALIDATION)
def system_processes_list(
    limit: int = 50,
    detailed: bool = False,
    state: AgentState | None = None,
) -> list[dict[str, Any]]:
    """Return a list of running processes (pid, name, status)."""
    limit = max(1, int(limit))
    entries: list[dict[str, Any]] = []
    base_attrs = ["pid", "name", "status"]
    detailed_attrs = ["cpu_percent", "memory_percent", "create_time", "username", "cmdline"]
    attrs = base_attrs + detailed_attrs if detailed else base_attrs
    try:
        iterator = psutil.process_iter(attrs)
    except (psutil.Error, OSError, PermissionError, SystemError):
        return []
    try:
        for process in iterator:
            info = process.info
            entry = {
                "pid": info.get("pid"),
                "name": info.get("name") or "",
                "status": info.get("status") or "",
            }
            if detailed:
                entry.update(
                    {
                        "cpu_percent": float(info.get("cpu_percent") or 0.0),
                        "memory_percent": float(info.get("memory_percent") or 0.0),
                        "create_time": float(info.get("create_time") or 0.0),
                        "username": info.get("username") or "",
                        "cmdline": info.get("cmdline") or [],
                    }
                )
            entries.append(entry)
            if len(entries) >= limit:
                break
    except (psutil.Error, OSError, PermissionError, SystemError):
        return []
    return entries

if __name__ == "__main__":
    # Example usage
    print("CPU Stats:", system_cpu_stats(per_cpu=True, interval=1.0))
    print("Memory Stats:", system_memory_stats())
    #print("Disk Usage:", system_disk_usage(path="/"))
    print("Current Process Snapshot:", system_process_snapshot())
    print("Top Processes by CPU:", system_processes_top(limit=3, sort_by="cpu"))
    print("Process List:", system_processes_list(limit=50, detailed=True))
