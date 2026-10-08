"""
Start Corvus:

    python -m corvus                       # workspace = current folder, confirm mode
    python -m corvus --workspace D:\\my-app
    python -m corvus --auto                # don't ask before risky actions
    python -m corvus --provider ollama --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import asyncio

from .cli import CorvusCLI


def main() -> None:
    parser = argparse.ArgumentParser(prog="corvus", description="Corvus, a CLI coding assistant")
    parser.add_argument("--workspace", default=".", help="project folder Corvus works in (default: current folder)")
    parser.add_argument("--config", default=None, help="path to servers.json (default: the one in the project root)")
    parser.add_argument("--provider", default=None, help="groq or ollama (default: CORVUS_PROVIDER in .env, else groq)")
    parser.add_argument("--model", default=None, help="model name for the provider")
    parser.add_argument("--auto", action="store_true", help="auto-execute mode: run tools without asking")
    args = parser.parse_args()

    cli = CorvusCLI(
        workspace=args.workspace,
        config_path=args.config,
        provider=args.provider,
        model=args.model,
        auto=args.auto,
    )
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
