"""
MCP server tools for Codur.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client, types

from codur.config import CodurConfig
from codur.graph.state import AgentState


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Cannot run MCP tools from within a running event loop")


def _resolve_config(config: CodurConfig | None, state: AgentState | None) -> CodurConfig:
    if config is not None:
        return config
    if state is not None and hasattr(state, "get_config"):
        cfg = state.get_config()
        if cfg is not None:
            return cfg
    raise ValueError("Config not available in tool state")


def _get_server(config: CodurConfig, server: str) -> StdioServerParameters:
    if server not in config.mcp_servers:
        raise ValueError(f"Unknown MCP server: {server}")
    cfg = config.mcp_servers[server]
    return StdioServerParameters(
        command=cfg.command,
        args=cfg.args,
        cwd=cfg.cwd,
        env=cfg.env,
    )


async def _with_session(config: CodurConfig, server: str, fn):
    params = _get_server(config, server)
    async with stdio_client(params) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        await session.initialize()
        return await fn(session)


def list_mcp_tools(
    server: str,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> list[dict]:
    config = _resolve_config(config, state)
    async def _list(session: ClientSession):
        result = await session.list_tools()
        return [tool.model_dump() for tool in result.tools]

    return _run(_with_session(config, server, _list))


def call_mcp_tool(
    server: str,
    tool: str,
    arguments: dict[str, Any] | None,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> dict:
    config = _resolve_config(config, state)
    async def _call(session: ClientSession):
        result = await session.call_tool(tool, arguments=arguments)
        return result.model_dump()

    return _run(_with_session(config, server, _call))


def list_mcp_resources(
    server: str,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> dict:
    config = _resolve_config(config, state)
    async def _list(session: ClientSession):
        result = await session.list_resources()
        return result.model_dump()

    return _run(_with_session(config, server, _list))


def list_mcp_resource_templates(
    server: str,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> dict:
    config = _resolve_config(config, state)
    async def _list(session: ClientSession):
        result = await session.list_resource_templates()
        return result.model_dump()

    return _run(_with_session(config, server, _list))


def read_mcp_resource(
    server: str,
    uri: str,
    config: CodurConfig | None = None,
    state: AgentState | None = None,
) -> dict:
    config = _resolve_config(config, state)
    async def _read(session: ClientSession):
        result = await session.read_resource(types.AnyUrl(uri))
        return result.model_dump()

    return _run(_with_session(config, server, _read))
