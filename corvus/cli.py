"""
Terminal interface (REPL) for Corvus, built with Rich.

Shows the model's answer as it streams, a panel for every tool call,
a spinner while something is running, and the confirmation prompt
before risky actions.

This is the first version for the check-in. The CLI owner can restyle
everything here without touching the agent loop.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mcp_tools import ConfirmationGate, MCPManager, ToolRegistry

from .agent import Agent
from .config import PROJECT_ROOT, Settings
from .providers import SUPPORTED_PROVIDERS, ProviderError, get_chat_model

BANNER = r"""
  _________  ______   ____  _______
 / ___/ __ \/ ___/ | / / / / / ___/
/ /__/ /_/ / /   | |/ / /_/ (__  )
\___/\____/_/    |___/\__,_/____/
"""

HELP = """[bold]Commands[/bold]
  /help                     show this help
  /tools                    list the tools loaded from each MCP server
  /mode confirm | auto      ask before risky actions, or run everything automatically
  /model groq | ollama \\[name]   switch LLM provider (and optionally the model)
  /clear                    start a fresh conversation
  /exit                     quit

Anything else is a task for Corvus, e.g. [italic]read README.md and summarize it[/italic]"""


class RichEvents:
    """Implements AgentEvents: turns loop events into terminal output."""

    def __init__(self, console: Console, gate: ConfirmationGate) -> None:
        self.console = console
        self.gate = gate
        self._status = None
        self._streaming = False

    # spinner helpers
    def _start_status(self, message: str) -> None:
        self._stop_status()
        self._status = self.console.status(message, spinner="dots")
        self._status.start()

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def end_stream(self) -> None:
        """Finish the current line of streamed text."""

        self._stop_status()
        if self._streaming:
            self.console.print()
            self._streaming = False

    # AgentEvents
    def on_thinking(self) -> None:
        self.end_stream()
        self._start_status("[cyan]Thinking…[/cyan]")

    def on_token(self, text: str) -> None:
        if not self._streaming:
            self._stop_status()
            self.console.print("[bold magenta]corvus ›[/bold magenta] ", end="")
            self._streaming = True
        self.console.print(text, end="", markup=False, highlight=False)

    def on_tool_start(self, name: str, server: str, args: dict[str, Any]) -> None:
        self.end_stream()
        args_text = json.dumps(args, indent=2, ensure_ascii=False)
        if len(args_text) > 1200:
            args_text = args_text[:1200] + "\n…"

        needs_ok = not self.gate.auto_execute and self.gate.requires_confirmation(name)
        tool_label = name.split("__", 1)[-1]
        self.console.print(
            Panel(
                Text(args_text),
                title=f"[bold]⚙ {tool_label}[/bold]  [dim]via {server}[/dim]",
                title_align="left",
                border_style="yellow" if needs_ok else "cyan",
                expand=False,
            )
        )
        # No spinner while waiting for the user's y/n answer.
        if not needs_ok:
            self._start_status(f"[cyan]Running {tool_label}…[/cyan]")

    def on_tool_end(self, name: str, preview: str, ok: bool) -> None:
        self._stop_status()
        lines = preview.strip().splitlines()
        short = "\n".join(lines[:6])
        if len(short) > 500:
            short = short[:500] + "…"
        if len(lines) > 6:
            short += f"\n… ({len(lines) - 6} more lines)"
        mark = Text("  ✓ " if ok else "  ✗ ", style="green" if ok else "red")
        self.console.print(mark + Text(short or "(no output)", style="dim"))

    def on_error(self, message: str) -> None:
        self.end_stream()
        self.console.print(Panel(Text(message), title="Error", border_style="red", expand=False))


class CorvusCLI:
    """Sets everything up and runs the read-eval-print loop."""

    def __init__(
        self,
        workspace: str | Path = ".",
        config_path: str | Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        auto: bool = False,
    ) -> None:
        self.console = Console()
        self.settings = Settings()
        self.workspace = Path(workspace).resolve()
        self.config_path = Path(config_path) if config_path else PROJECT_ROOT / "servers.json"
        self.provider = (provider or self.settings.provider).lower()
        self.model_name = model or self.settings.default_model(self.provider)

        self.gate = ConfirmationGate(auto_execute=auto, input_fn=self._ask_permission)
        self.registry = ToolRegistry(
            MCPManager(self.config_path, self.workspace),
            workspace=self.workspace,
            confirmation_gate=self.gate,
        )
        self.events = RichEvents(self.console, self.gate)
        self.agent: Agent | None = None

    def _ask_permission(self, _prompt: str) -> str:
        """Called by the ConfirmationGate. The tool panel already shows the details."""

        return self.console.input(Text("  Allow this action? [y/N] ", style="bold yellow"))

    # ---- startup -----------------------------------------------------------

    async def start(self) -> None:
        self.console.print(Text(BANNER, style="bold magenta"))
        self.console.print("[dim]  an autonomous coding assistant · type /help for commands[/dim]\n")

        with self.console.status("[cyan]Connecting to MCP servers…[/cyan]"):
            await self.registry.initialize()

        model = get_chat_model(self.provider, self.model_name, self.settings)
        self.agent = Agent(
            self.registry,
            model,
            self.events,
            self.workspace,
            self.settings,
            context_budget=self.settings.context_budget(self.provider),
        )
        self._print_status()

    def _print_status(self) -> None:
        assert self.agent is not None
        per_server = Counter(
            name.split("__", 1)[0] if "__" in name else "local"
            for name in self.agent.tool_names
        )
        servers = ", ".join(f"{s} ({n})" for s, n in sorted(per_server.items()))
        failed = getattr(self.registry.mcp, "failed_servers", {}) or {}

        self.console.print(f"  [bold]workspace[/bold]  {self.workspace}")
        self.console.print(f"  [bold]model[/bold]      {self.provider} · {self.model_name}")
        self.console.print(f"  [bold]mode[/bold]       {self._mode_label()}")
        self.console.print(f"  [bold]tools[/bold]      {servers}")
        for name, error in failed.items():
            self.console.print(f"  [red]✗ {name} failed to start:[/red] {error}")
        self.console.print()

    def _mode_label(self) -> str:
        if self.gate.auto_execute:
            return "[red]auto[/red] (runs everything without asking)"
        return "[green]confirm[/green] (asks before writing files or running commands)"

    # ---- slash commands ------------------------------------------------------

    def _handle_command(self, line: str) -> bool:
        """Run a /command. Returns False when the user wants to quit."""

        assert self.agent is not None
        parts = line.split()
        command, args = parts[0].lower(), parts[1:]

        if command in ("/exit", "/quit"):
            return False

        if command == "/help":
            self.console.print(HELP)

        elif command == "/tools":
            table = Table(title="Loaded tools", show_lines=False)
            table.add_column("server", style="cyan")
            table.add_column("tool")
            table.add_column("asks first?", justify="center")
            for tool in sorted(self.agent.tools, key=lambda t: t["name"]):
                name = tool["name"]
                server, _, short = name.rpartition("__")
                risky = self.gate.requires_confirmation(name)
                table.add_row(server or "local", short, "[yellow]yes[/yellow]" if risky else "")
            self.console.print(table)

        elif command == "/mode":
            if args and args[0] in ("auto", "confirm"):
                self.gate.auto_execute = args[0] == "auto"
            else:
                self.console.print("[dim]usage: /mode confirm  or  /mode auto[/dim]")
            self.console.print(f"  mode: {self._mode_label()}")

        elif command == "/model":
            if not args or args[0] not in SUPPORTED_PROVIDERS:
                self.console.print(
                    f"[dim]usage: /model groq|ollama \\[model-name]   "
                    f"(now: {self.provider} · {self.model_name})[/dim]"
                )
            else:
                provider = args[0]
                name = args[1] if len(args) > 1 else self.settings.default_model(provider)
                try:
                    self.agent.set_model(
                        get_chat_model(provider, name, self.settings),
                        self.settings.context_budget(provider),
                    )
                    self.provider, self.model_name = provider, name
                    self.console.print(f"  model: {provider} · {name}")
                except ProviderError as exc:
                    self.events.on_error(str(exc))

        elif command == "/clear":
            self.agent.reset()
            self.console.print("  [dim]conversation cleared[/dim]")

        else:
            self.console.print(f"[dim]unknown command {command}, try /help[/dim]")

        return True

    # ---- main loop -------------------------------------------------------------

    async def run(self) -> None:
        try:
            await self.start()
            assert self.agent is not None

            while True:
                try:
                    line = self.console.input(Text("you › ", style="bold green")).strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not line:
                    continue
                if line.startswith("/"):
                    if not self._handle_command(line):
                        break
                    continue

                await self.agent.run(line)
                self.events.end_stream()
                self.console.print()

        except ProviderError as exc:
            self.events.on_error(str(exc))
        finally:
            self.events.end_stream()
            await self.registry.close()
            self.console.print("[dim]bye[/dim]")
