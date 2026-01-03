"""
Pandoc document conversion tools for Codur.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from codur.constants import TaskType
from codur.graph.state import AgentState
from codur.graph.state_operations import get_config
from codur.tools.tool_annotations import (
    ToolContext,
    ToolSideEffect,
    tool_contexts,
    tool_scenarios,
    tool_side_effects,
)
from codur.utils.path_utils import resolve_path, resolve_root
from codur.utils.validation import validate_file_access


_DEFAULT_EXTENSIONS = {
    "markdown": "md",
    "gfm": "md",
    "html": "html",
    "pdf": "pdf",
    "docx": "docx",
    "rst": "rst",
    "txt": "txt",
}


@tool_side_effects(ToolSideEffect.FILE_MUTATION)
@tool_contexts(ToolContext.FILESYSTEM)
@tool_scenarios(TaskType.FILE_OPERATION, TaskType.EXPLANATION, TaskType.DOCUMENTATION)
def convert_document(
    input_path: str,
    output_format: str,
    output_path: str | None = None,
    input_format: str | None = None,
    extra_args: list[str] | None = None,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """
    Convert a document using pandoc.
    """
    if shutil.which("pandoc") is None:
        raise FileNotFoundError("pandoc is not available on PATH")

    source = resolve_path(input_path, root, allow_outside_root=allow_outside_root)
    validate_file_access(
        source,
        resolve_root(root),
        get_config(state),
        operation="read",
        allow_outside_root=allow_outside_root,
    )

    if output_path is None:
        ext = _DEFAULT_EXTENSIONS.get(output_format, output_format)
        output = source.with_suffix(f".{ext}")
    else:
        output = resolve_path(output_path, root, allow_outside_root=allow_outside_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["pandoc", str(source)]
    if input_format:
        cmd += ["-f", input_format]
    cmd += ["-t", output_format, "-o", str(output)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"pandoc failed: {stderr}")

    return {
        "input": str(source),
        "output": str(output),
        "format": output_format,
        "command": " ".join(cmd),
    }
