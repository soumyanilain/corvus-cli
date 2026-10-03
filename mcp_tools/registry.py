from __future__ import annotations

from pathlib import Path
from typing import Any

from .confirmation import ConfirmationGate
from .local_tools import grep_code, run_command
from .manager import MCPManager


class ToolRegistry:
    """
    Combines:
    - MCP-discovered tools
    - run_command
    - grep_code
    - confirmation / auto-execution behavior
    """

    def __init__(
        self,
        mcp_manager: MCPManager,
        *,
        workspace: str | Path = ".",
        confirmation_gate: ConfirmationGate | None = None,
    ) -> None:

        self.mcp = mcp_manager

        self.workspace = Path(
            workspace
        ).resolve()

        self.confirmation_gate = (
            confirmation_gate
            or ConfirmationGate()
        )

    async def initialize(
        self,
    ) -> None:

        await self.mcp.connect_all()

    def tool_schemas(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return all tools in one common format
        that the agent/provider layer can use.
        """

        schemas: list[
            dict[str, Any]
        ] = [
            {
                "name": "run_command",
                "description": (
                    "Run a shell command inside "
                    "the current project workspace."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string"
                        },
                        "cwd": {
                            "type": "string",
                            "default": ".",
                        },
                        "timeout": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 600,
                            "default": 120,
                        },
                    },
                    "required": [
                        "command"
                    ],
                },
            },
            {
                "name": "grep_code",
                "description": (
                    "Search the local codebase "
                    "for text or a regular expression."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string"
                        },
                        "path": {
                            "type": "string",
                            "default": ".",
                        },
                        "regex": {
                            "type": "boolean",
                            "default": False,
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "default": False,
                        },
                        "max_matches": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "default": 100,
                        },
                    },
                    "required": [
                        "pattern"
                    ],
                },
            },
        ]

        for tool in self.mcp.registered_tools():

            schemas.append(
                {
                    "name": tool.public_name,
                    "description": (
                        tool.description
                    ),
                    "input_schema": (
                        tool.input_schema
                    ),
                }
            )

        return schemas

    async def execute(
        self,
        tool_name: str,
        arguments: dict[
            str,
            Any,
        ] | None = None,
    ) -> dict[str, Any]:

        arguments = arguments or {}

        approved = (
            await self.confirmation_gate.approve(
                tool_name,
                arguments,
            )
        )

        if not approved:
            return {
                "ok": False,
                "cancelled": True,
                "tool": tool_name,
                "message": (
                    "User denied tool execution."
                ),
            }

        if tool_name == "run_command":

            return await run_command(
                arguments["command"],
                workspace=self.workspace,
                cwd=arguments.get(
                    "cwd",
                    ".",
                ),
                timeout=int(
                    arguments.get(
                        "timeout",
                        120,
                    )
                ),
            )

        if tool_name == "grep_code":

            return await grep_code(
                arguments["pattern"],
                workspace=self.workspace,
                path=arguments.get(
                    "path",
                    ".",
                ),
                regex=bool(
                    arguments.get(
                        "regex",
                        False,
                    )
                ),
                case_sensitive=bool(
                    arguments.get(
                        "case_sensitive",
                        False,
                    )
                ),
                max_matches=int(
                    arguments.get(
                        "max_matches",
                        100,
                    )
                ),
            )

        if self.mcp.has_tool(
            tool_name
        ):

            return await self.mcp.call_tool(
                tool_name,
                arguments,
            )

        raise KeyError(
            f"Unknown tool: {tool_name}"
        )

    async def close(
        self,
    ) -> None:

        await self.mcp.close()