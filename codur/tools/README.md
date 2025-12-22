# Codur Tools

Tools are plain Python functions that the tool runner can call. Keep them simple,
safe, and easy to serialize.

## How To Add A Tool

1) Create a module in `codur/tools/` with your function(s).
2) Export the function in `codur/tools/__init__.py` and add it to `__all__`.
3) Register the tool in `codur/graph/nodes/tools.py` so it can be executed.
4) Add a short docstring summary. It shows up in the tool directory listing.
5) If the tool touches files, use `resolve_path()` / `resolve_root()` and honor
   `allow_outside_root`.
6) Return JSON-serializable data and keep outputs reasonably small.
7) Add any required dependencies to `pyproject.toml`.

## Example

```python
from __future__ import annotations

from pathlib import Path
from codur.graph.state import AgentState
from codur.utils.path_utils import resolve_path


def example_tool(
    path: str,
    root: str | Path | None = None,
    allow_outside_root: bool = False,
    state: AgentState | None = None,
) -> dict:
    """One-line summary shown in tool listings."""
    target = resolve_path(path, root, allow_outside_root=allow_outside_root)
    return {"path": str(target)}
```
