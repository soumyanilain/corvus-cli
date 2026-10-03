from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ConfirmationGate:
    """
    Handles confirmation before tools perform risky actions.

    auto_execute=False:
        Ask the user before running risky tools.

    auto_execute=True:
        Run tools automatically without asking.
    """

    auto_execute: bool = False
    input_fn: Callable[[str], str] = input

    risky_keywords: tuple[str, ...] = field(
        default=(
            "run_command",
            "write",
            "edit",
            "delete",
            "remove",
            "move",
            "rename",
            "create_directory",
            "mkdir",
            "replace",
            "patch",
        )
    )

    def requires_confirmation(self, tool_name: str) -> bool:
        name = tool_name.lower()

        return any(
            keyword in name
            for keyword in self.risky_keywords
        )

    async def approve(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:

        if self.auto_execute:
            return True

        if not self.requires_confirmation(tool_name):
            return True

        details = json.dumps(
            arguments,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        prompt = (
            f"\nTool '{tool_name}' wants to execute:\n"
            f"{details}\n"
            "Allow this action? [y/N]: "
        )

        answer = await asyncio.to_thread(
            self.input_fn,
            prompt,
        )

        return answer.strip().lower() in {
            "y",
            "yes",
        }