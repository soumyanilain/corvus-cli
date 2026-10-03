from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .config import load_server_config


@dataclass
class RegisteredMCPTool:
    """
    Information about one tool loaded from an MCP server.
    """

    public_name: str
    server_name: str
    original_name: str
    description: str
    input_schema: dict[str, Any]


class MCPManager:
    """
    Connects to multiple MCP servers and dynamically
    discovers the tools they provide.

    MCP tools are exposed using names like:

        filesystem__read_file
        tavily__tavily_search

    This prevents tool-name conflicts between servers.
    """

    def __init__(
        self,
        config_path: str | Path = "servers.json",
        workspace: str | Path = ".",
    ) -> None:
        self.config_path = Path(config_path)

        self.workspace = Path(
            workspace
        ).resolve()

        self._stack = AsyncExitStack()

        self._clients: dict[str, Client] = {}

        self._tools: dict[
            str,
            RegisteredMCPTool,
        ] = {}

        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def has_tool(
        self,
        public_name: str,
    ) -> bool:
        return public_name in self._tools

    def registered_tools(
        self,
    ) -> list[RegisteredMCPTool]:
        return list(
            self._tools.values()
        )

    async def connect_all(
        self,
    ) -> list[RegisteredMCPTool]:

        if self._connected:
            return self.registered_tools()

        configs = load_server_config(
            self.config_path,
            self.workspace,
        )

        for server_name, config in configs.items():

            if not config.get(
                "enabled",
                True,
            ):
                continue

            await self._connect_server(
                server_name,
                config,
            )

        self._connected = True

        return self.registered_tools()

    async def _connect_server(
        self,
        server_name: str,
        config: dict[str, Any],
    ) -> None:

        transport = config.get(
            "transport",
            "stdio",
        ).lower()

        if transport == "stdio":

            command = config.get(
                "command"
            )

            if not command:
                raise ValueError(
                    f"MCP server "
                    f"'{server_name}' "
                    f"is missing 'command'."
                )

            target = StdioServerParameters(
                command=command,
                args=config.get(
                    "args",
                    [],
                ),
                env=config.get(
                    "env"
                ),
            )

        elif transport in {
            "http",
            "streamable-http",
            "streamable_http",
        }:

            url = config.get(
                "url"
            )

            if not url:
                raise ValueError(
                    f"MCP server "
                    f"'{server_name}' "
                    f"is missing 'url'."
                )

            target = url

        else:
            raise ValueError(
                f"Unsupported MCP transport "
                f"'{transport}' "
                f"for server "
                f"'{server_name}'."
            )

        client = await self._stack.enter_async_context(
            Client(target)
        )

        self._clients[
            server_name
        ] = client

        page = await client.list_tools()

        tools = list(
            page.tools
        )

        cursor = page.next_cursor

        while cursor:

            page = await client.list_tools(
                cursor=cursor
            )

            tools.extend(
                page.tools
            )

            cursor = page.next_cursor

        for tool in tools:

            original_name = tool.name

            public_name = (
                f"{server_name}"
                f"__"
                f"{original_name}"
            )

            if public_name in self._tools:
                raise RuntimeError(
                    f"Duplicate MCP tool name: "
                    f"{public_name}"
                )

            self._tools[
                public_name
            ] = RegisteredMCPTool(
                public_name=public_name,
                server_name=server_name,
                original_name=original_name,
                description=(
                    tool.description or ""
                ),
                input_schema=(
                    tool.input_schema
                    or {
                        "type": "object",
                        "properties": {},
                    }
                ),
            )

    async def call_tool(
        self,
        public_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        if not self._connected:
            raise RuntimeError(
                "MCPManager is not connected. "
                "Call connect_all() first."
            )

        tool = self._tools.get(
            public_name
        )

        if tool is None:
            raise KeyError(
                f"Unknown MCP tool: "
                f"{public_name}"
            )

        client = self._clients[
            tool.server_name
        ]

        result = await client.call_tool(
            tool.original_name,
            arguments or {},
        )

        if hasattr(
            result,
            "model_dump",
        ):
            return result.model_dump(
                by_alias=True,
                mode="json",
                exclude_none=True,
            )

        return {
            "content": str(result)
        }

    async def close(
        self,
    ) -> None:

        await self._stack.aclose()

        self._clients.clear()

        self._tools.clear()

        self._connected = False

    async def __aenter__(
        self,
    ) -> "MCPManager":

        await self.connect_all()

        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:

        await self.close()