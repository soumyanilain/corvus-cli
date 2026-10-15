"""
Compare LLMs on the same coding task, with no human in the loop.

    python -m eval.llm_compare
    python -m eval.llm_compare --models groq:openai/gpt-oss-20b ollama:llama3.1:8b

Each model's run is saved in eval/results/runs/, and the summary table covers
every saved run, so models can be run in separate invocations.

For each model:
  1. Copy eval/sample_project (a small FastAPI app with a bug) to a fresh folder.
  2. Run Corvus on TASK in auto mode (no confirmation prompts), recording every
     model call and tool call.
  3. Grade the result: run the tests the agent wrote, then our hidden grading
     tests (eval/grading/test_grading.py), which the model never sees.

Results go to eval/results/llm_compare.json and eval/results/llm_compare.md,
plus a step-by-step log per model in eval/results/runs/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from corvus.agent import Agent
from corvus.config import PROJECT_ROOT, Settings
from corvus.providers import get_chat_model
from mcp_tools import ConfirmationGate, MCPManager, ToolRegistry

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS_DIR = PROJECT_ROOT / "data" / "llm_runs"

DEFAULT_MODELS = ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b", "groq:qwen/qwen3.8-27b"]

TASK = """This folder is a small FastAPI app (app.py) with tests in test_app.py. Make these changes:
1. Add a POST /books endpoint that accepts a JSON body validated with a Pydantic model:
   title (non-empty string), author (string), year (integer between 1450 and 2100).
   Store the new book with the next free id and return it with status code 201.
2. GET /books/{book_id} crashes with a 500 error when the id doesn't exist. Make it return a 404 error instead.
Check the FastAPI docs for how to do these. Add tests for both changes to test_app.py,
then run the tests with `python -m pytest -q` and fix anything that fails."""


class RecordingEvents:
    """AgentEvents that log everything for the report and print a short progress line."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.model_calls = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.notices: list[str] = []
        self._pending: dict[str, Any] | None = None

    def on_thinking(self) -> None:
        self.model_calls += 1

    def on_token(self, text: str) -> None:
        pass

    def on_tool_start(self, name: str, server: str, args: dict[str, Any]) -> None:
        self._pending = {"tool": name, "server": server, "args": args, "started": time.time()}
        print(f"  [{self.label}] {server:<10} {name.split('__')[-1]}", flush=True)

    def on_tool_end(self, name: str, preview: str, ok: bool) -> None:
        call = self._pending or {"tool": name, "server": "?", "args": {}, "started": time.time()}
        call.update(ok=ok, seconds=round(time.time() - call.pop("started"), 2), result=preview[:300])
        self.tool_calls.append(call)
        self._pending = None

    def on_notice(self, message: str) -> None:
        self.notices.append(message)
        print(f"  [{self.label}] note: {message}", flush=True)

    def on_error(self, message: str) -> None:
        self.errors.append(message[:500])
        print(f"  [{self.label}] ERROR {message[:150]}", flush=True)


def run_pytest(workspace: Path, target: str) -> dict[str, Any]:
    """Run pytest on one file and return pass/fail counts."""

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider"],
        cwd=workspace, capture_output=True, text=True, timeout=180,
    )
    out = proc.stdout + proc.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    return {"passed": passed, "failed": failed + errors, "exit_code": proc.returncode,
            "tail": out.strip().splitlines()[-3:]}


def token_usage(messages: list) -> dict[str, int] | None:
    """Sum token usage reported by the provider, if it reports any."""

    total = Counter()
    for m in messages:
        usage = getattr(m, "usage_metadata", None) if isinstance(m, AIMessage) else None
        if usage:
            total["input"] += usage.get("input_tokens", 0)
            total["output"] += usage.get("output_tokens", 0)
    return dict(total) or None


async def run_one(spec: str, settings: Settings, servers: Path) -> dict[str, Any]:
    provider, _, model_name = spec.partition(":")
    label = model_name.split("/")[-1]

    workspace = RUNS_DIR / re.sub(r"[^\w.-]", "_", spec)
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(HERE / "sample_project", workspace)

    events = RecordingEvents(label)
    registry = ToolRegistry(
        MCPManager(servers, workspace),
        workspace=workspace,
        confirmation_gate=ConfirmationGate(auto_execute=True),
    )
    print(f"\n=== {spec} ===")
    await registry.initialize()
    try:
        agent = Agent(
            registry,
            get_chat_model(provider, model_name, settings),
            events,
            workspace,
            settings,
            context_budget=settings.context_budget(provider),
        )
        started = time.time()
        answer = await agent.run(TASK)
        seconds = round(time.time() - started, 1)
        hit_step_limit = any("step limit" in str(m.content) for m in agent.messages[-1:])
        tokens = token_usage(agent.messages)
    finally:
        await registry.close()

    agent_tests = run_pytest(workspace, "test_app.py")
    shutil.copy(HERE / "grading" / "test_grading.py", workspace / "_grading_test.py")
    grading = run_pytest(workspace, "_grading_test.py")

    by_server = Counter(c["server"] for c in events.tool_calls)
    result = {
        "model": spec,
        "seconds": seconds,
        "model_calls": events.model_calls,
        "tool_calls": len(events.tool_calls),
        "failed_tool_calls": sum(not c["ok"] for c in events.tool_calls),
        "tool_calls_by_server": dict(by_server),
        "tools_used": dict(Counter(c["tool"].split("__")[-1] for c in events.tool_calls)),
        "hit_step_limit": hit_step_limit,
        "errors": events.errors,
        "recoveries": events.notices,
        "tokens": tokens,
        "agent_tests": agent_tests,
        "grading": grading,
        "grading_score": f"{grading['passed']}/{grading['passed'] + grading['failed']}",
        "final_answer": answer[:1500],
    }

    runs = RESULTS / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{workspace.name}.json").write_text(
        json.dumps({**result, "steps": events.tool_calls}, indent=2, default=str), encoding="utf-8"
    )
    print(f"  -> grading {result['grading_score']}, {result['tool_calls']} tool calls, {seconds}s")
    return result


def all_runs() -> list[dict[str, Any]]:
    """Every saved run (one per model), so separate invocations add up to one table."""

    results = []
    for path in sorted((RESULTS / "runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        run.pop("steps", None)
        # A model the API key can't use isn't part of the comparison.
        if any("model_not_found" in e for e in run.get("errors", [])):
            continue
        results.append(run)
    return results


def write_report(results: list[dict[str, Any]]) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "llm_compare.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    cols = [r["model"] for r in results]
    def row(label: str, fn) -> str:
        return f"| {label} | " + " | ".join(str(fn(r)) for r in results) + " |"

    md = [
        "# LLM comparison on the same coding task",
        "",
        "Task (identical for every model, run in auto mode on a fresh copy of eval/sample_project):",
        "",
        "> " + TASK.replace("\n", "\n> "),
        "",
        "| | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
        row("Hidden grading tests passed", lambda r: r["grading_score"]),
        row("Used the docs search (docs-rag)", lambda r: "yes" if r["tool_calls_by_server"].get("docs-rag") else "no"),
        row("Agent's own tests (pass/fail)", lambda r: f"{r['agent_tests']['passed']}/{r['agent_tests']['failed']}"),
        row("Time (s)", lambda r: r["seconds"]),
        row("Model calls", lambda r: r["model_calls"]),
        row("Tool calls (failed)", lambda r: f"{r['tool_calls']} ({r['failed_tool_calls']})"),
        row("Calls per server", lambda r: ", ".join(f"{k} {v}" for k, v in r["tool_calls_by_server"].items())),
        row("Tokens in/out", lambda r: f"{r['tokens']['input']}/{r['tokens']['output']}" if r["tokens"] else "n/a"),
        row("Hit step limit", lambda r: "yes" if r["hit_step_limit"] else "no"),
        row("Automatic recoveries", lambda r: len(r.get("recoveries", []))),
        row("Errors that stopped the task", lambda r: len(r["errors"])),
    ]
    (RESULTS / "llm_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS / 'llm_compare.md'}")


async def main_async(models: list[str], pause: int, servers: Path) -> None:
    settings = Settings()
    results = []
    for i, spec in enumerate(models):
        if i and pause:
            print(f"\n(waiting {pause}s so the next run starts with a fresh per-minute token allowance)")
            await asyncio.sleep(pause)
        results.append(await run_one(spec, settings, servers))
    write_report(all_runs())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLMs on the same Corvus coding task.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="provider:model entries, e.g. groq:openai/gpt-oss-120b ollama:llama3.1:8b")
    parser.add_argument("--pause", type=int, default=60, help="seconds to wait between models")
    parser.add_argument("--servers", type=Path, default=PROJECT_ROOT / "servers.json", help="MCP server config")
    args = parser.parse_args()
    asyncio.run(main_async(args.models, args.pause, args.servers))


if __name__ == "__main__":
    main()
