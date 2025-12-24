"""Tests for MCP tool wrappers."""

import pytest

mcp = pytest.importorskip("mcp")

from codur.tools.mcp_tools import (
    list_mcp_tools,
    call_mcp_tool,
    list_mcp_resources,
    list_mcp_resource_templates,
    read_mcp_resource,
)


@pytest.mark.parametrize(
    "func,args",
    [
        (list_mcp_tools, ("server",)),
        (call_mcp_tool, ("server", "tool", None)),
        (list_mcp_resources, ("server",)),
        (list_mcp_resource_templates, ("server",)),
        (read_mcp_resource, ("server", "mcp://resource")),
    ],
)
def test_mcp_tools_require_config(func, args):
    with pytest.raises(ValueError, match="Config not available"):
        func(*args)
