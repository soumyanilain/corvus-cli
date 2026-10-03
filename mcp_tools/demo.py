from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .confirmation import ConfirmationGate
from .manager import MCPManager
from .registry import ToolRegistry


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Sogol's MCP + Tools implementation."
    )

    parser.add_argument(
        "--workspace",
        default=".",
    )

    parser.add_argument(
        "--config",
        default="servers.json",
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Auto-execute risky tools "
            "without confirmation."
        ),
    )

    args = parser.parse_args()

    workspace = Path(
        args.workspace
    ).resolve()

    manager = MCPManager(
        config_path=args.config,
        workspace=workspace,
    )

    registry = ToolRegistry(
        manager,
        workspace=workspace,
        confirmation_gate=ConfirmationGate(
            auto_execute=args.auto
        ),
    )

    try:
        print(
            f"Workspace: {workspace}"
        )

        print(
            "Connecting to MCP servers..."
        )

        await registry.initialize()

        tools = registry.tool_schemas()

        print(
            f"\nLoaded {len(tools)} tools:\n"
        )

        for tool in tools:
            print(
                f" - {tool['name']}: "
                f"{tool.get('description', '')}"
            )

        print(
            "\nLocal grep_code smoke test:"
        )

        result = await registry.execute(
            "grep_code",
            {
                "pattern": "class ",
                "path": ".",
                "max_matches": 5,
            },
        )

        for match in result.get(
            "matches",
            [],
        ):
            print(
                f"   {match['path']}:"
                f"{match['line']}  "
                f"{match['text']}"
            )

        print(
            "\nMCP + Tools setup is working."
        )

    finally:
        await registry.close()


if __name__ == "__main__":
    asyncio.run(main())