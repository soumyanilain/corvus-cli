"""
The agentic loop: reason -> act -> observe -> repeat.

    messages = [system prompt, user task]
    loop:
        ask the model (streaming), with every tool's schema attached
        if the model asks for no tools -> it's done, stop
        otherwise run each tool it asked for (via the ToolRegistry),
        add the results to the conversation, and ask the model again

The loop doesn't print anything itself. It reports what is happening through
an AgentEvents object, and the CLI decides how to show it. That keeps the
loop testable and lets the CLI change without touching this file.

Context management: every request re-sends the whole conversation plus all
tool definitions. Groq's free tier rejects any single request over 8,000
tokens, so before each model call _fit_context() shrinks old tool outputs
and, if needed, drops earlier tasks to stay under a token budget.
"""

from __future__ import annotations

import copy
import json
import platform
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import message_chunk_to_message

from mcp_tools import ToolRegistry

from .config import DEFAULT_TREE_EXCLUDES, HIDDEN_TOOLS, Settings


class AgentEvents(Protocol):
    """What the loop tells the outside world. The CLI implements this."""

    def on_thinking(self) -> None: ...
    def on_token(self, text: str) -> None: ...
    def on_tool_start(self, name: str, server: str, args: dict[str, Any]) -> None: ...
    def on_tool_end(self, name: str, preview: str, ok: bool) -> None: ...
    def on_error(self, message: str) -> None: ...


SYSTEM_PROMPT = """You are Corvus, an autonomous coding assistant running in the user's terminal.
You complete tasks by calling tools, checking the results, and continuing until the task is done.

Workspace (project root): {workspace}
Operating system: {os_name}. Shell commands run with {shell}.
Connected tool servers: {servers}

How to work:
- Look before you change anything. Read the relevant files instead of guessing what they contain.
- Filesystem paths can be absolute ({example_path}) or relative to the workspace.
- Use list_directory to see a folder. Read a file once, without head or tail. If a result is cut,
  it ends with a note saying exactly how to read the rest: follow it. Don't re-read a file you already have.
- grep_code searches inside files. filesystem__search_files finds files by name.
- run_command runs shell commands (tests, git, pip, python) inside the workspace.
- Use tavily__tavily_search (max_results 3) for anything you need from the web: docs, error messages, latest versions.
- After every action, check the result. If something failed, read the error and try a different approach.
- If the user declines an action, don't try it again. Choose another approach or ask the user.
- When the task is finished, reply with a short summary of what you did, without calling any tools.
"""

# Replaces old tool outputs when the conversation gets too big.
STUB = "[older tool output removed to keep the conversation within the model's size limit]"


def _text_of(content: Any) -> str:
    """Message content can be a string or a list of parts; return the text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def format_tool_result(result: Any) -> tuple[str, bool]:
    """
    Turn a ToolRegistry result (a dict) into plain text for the model,
    plus a flag saying whether it succeeded.
    """

    if not isinstance(result, dict):
        return str(result), True

    # The user said "no" at the confirmation prompt.
    if result.get("cancelled"):
        return (
            "The user declined this action. Do not retry it. "
            "Choose a different approach or ask the user.",
            False,
        )

    # Errors returned by the registry or raised by a tool.
    if "error" in result and result.get("ok") is False:
        return f"Error: {result['error']}", False

    # MCP tool results: {"content": [{"type": "text", "text": ...}], "isError": bool}
    if isinstance(result.get("content"), list):
        parts = []
        for item in result["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                parts.append(f"[{item.get('type', 'non-text')} content omitted]")
        return "\n".join(parts), not result.get("isError", False)

    # run_command: success means exit code 0 and no timeout.
    if "returncode" in result:
        ok = result.get("returncode") == 0 and not result.get("timed_out")
        return json.dumps(result, indent=2, ensure_ascii=False), ok

    # Anything else (e.g. grep_code): send the JSON.
    return json.dumps(result, indent=2, ensure_ascii=False, default=str), True


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = len(text) - limit
    return (
        text[:limit]
        + f"\n... [output cut: {cut} more characters not shown. Don't repeat the same call; "
        "use head/tail, a narrower path, or grep_code to see other parts.]"
    )


def _truncate_file_read(text: str, limit: int, args: dict[str, Any]) -> str:
    """
    Cut a long file at a line boundary and tell the model exactly how to read
    the rest. The filesystem server only supports head/tail (no offset), so a
    vague "output was cut" note makes models retry the same read forever.
    """

    lines = text.splitlines(keepends=True)
    shown, size = 0, 0
    for line in lines:
        if size + len(line) > limit:
            break
        size += len(line)
        shown += 1

    total = len(lines)
    if shown == total:
        return text
    if shown == 0:  # one giant line; fall back to a plain cut
        return _truncate(text, limit)

    rest = total - shown
    head = args.get("head")
    if isinstance(head, (int, float)) and total >= head:
        # The head limit itself cut the file, so the true end is unknown.
        hint = "To see more, read the file again without head."
    else:
        # We have the whole file (or its tail), so tail=rest gives exactly what's missing.
        hint = (
            f"To read the remaining {rest} lines, call filesystem__read_text_file "
            f"with the same path and tail={rest} (and no head)."
        )
    return (
        "".join(lines[:shown])
        + f"\n... [output cut: showing the first {shown} of {total} lines. {hint}]"
    )


def _clean_schema(schema: Any, max_desc: int = 150) -> Any:
    """
    Make a tool's input schema smaller and provider-friendly:
    drop "$schema" (some providers reject it) and shorten long
    parameter descriptions (they are re-sent on every request).
    """

    if isinstance(schema, dict):
        cleaned = {}
        for key, value in schema.items():
            if key == "$schema":
                continue
            if key == "description" and isinstance(value, str):
                cleaned[key] = value[:max_desc]
            else:
                cleaned[key] = _clean_schema(value, max_desc)
        return cleaned
    if isinstance(schema, list):
        return [_clean_schema(item, max_desc) for item in schema]
    return copy.deepcopy(schema)


class Agent:
    """Runs tasks with an LLM and the tools in a ToolRegistry."""

    def __init__(
        self,
        registry: ToolRegistry,
        model: BaseChatModel,
        events: AgentEvents,
        workspace: str | Path,
        settings: Settings | None = None,
        context_budget: int | None = None,
    ) -> None:
        self.registry = registry
        self.events = events
        self.workspace = Path(workspace).resolve()
        self.settings = settings or Settings()

        self.tools = self._visible_tools()
        self.tool_names = {tool["name"] for tool in self.tools}
        self._tools_chars = len(json.dumps(self.tools))
        self.messages: list[BaseMessage] = [SystemMessage(self._system_prompt())]

        # Read-only calls already made in this task (to stop the model repeating them).
        self._seen_calls: set[str] = set()
        self._call_keys: dict[str, str] = {}  # tool_call_id -> call key
        # How many times each file was read in this task (stops head/tail loops).
        self._file_reads: dict[str, int] = {}

        self.set_model(model, context_budget or self.settings.context_budget("groq"))

    # ---- setup -----------------------------------------------------------

    def _visible_tools(self) -> list[dict[str, Any]]:
        """All tools from the registry, minus the hidden ones, in a provider-friendly shape."""

        tools = []
        for tool in self.registry.tool_schemas():
            if tool["name"] in HIDDEN_TOOLS:
                continue
            tools.append(
                {
                    "name": tool["name"],
                    "description": (tool.get("description") or "")[:300],
                    "input_schema": _clean_schema(tool["input_schema"]),
                }
            )
        return tools

    def _system_prompt(self) -> str:
        is_windows = platform.system() == "Windows"
        servers = sorted(
            {name.split("__", 1)[0] for name in self.tool_names if "__" in name}
        )
        return SYSTEM_PROMPT.format(
            workspace=self.workspace,
            os_name=platform.system(),
            shell="cmd.exe" if is_windows else "/bin/sh",
            servers=", ".join(servers) or "none",
            example_path=self.workspace / "README.md",
        )

    def set_model(self, model: BaseChatModel, context_budget: int) -> None:
        """Switch LLM (used by /model). Conversation history is kept."""

        self.model = model
        self.llm = model.bind_tools(self.tools)
        self.context_budget = context_budget

    def reset(self) -> None:
        """Forget the conversation (used by /clear)."""

        self.messages = [SystemMessage(self._system_prompt())]
        self._seen_calls.clear()
        self._call_keys.clear()
        self._file_reads.clear()

    # ---- context size ----------------------------------------------------------

    def estimate_tokens(self) -> int:
        """
        Rough size of the next request in tokens: tool definitions + all messages.
        Uses 4 characters per token, which over-estimates a little (safe side).
        """

        chars = self._tools_chars
        for message in self.messages:
            chars += len(_text_of(message.content)) + 20
            if isinstance(message, AIMessage) and message.tool_calls:
                chars += len(json.dumps([call.get("args") for call in message.tool_calls]))
                chars += 60 * len(message.tool_calls)
        return chars // 4

    def _fit_context(self) -> None:
        """Shrink the conversation until the next request fits the token budget."""

        if self.estimate_tokens() <= self.context_budget:
            return

        # 1. Replace old tool outputs with a short note, oldest first.
        #    Keep the newest results: the model is working with them right now.
        last_ai = max(
            (i for i, m in enumerate(self.messages) if isinstance(m, AIMessage)),
            default=len(self.messages),
        )
        for i, message in enumerate(self.messages[:last_ai]):
            if isinstance(message, ToolMessage) and message.content != STUB:
                self.messages[i] = ToolMessage(content=STUB, tool_call_id=message.tool_call_id)
                # The model may legitimately need that output again now.
                self._seen_calls.discard(self._call_keys.get(message.tool_call_id, ""))
                self._file_reads.clear()
                if self.estimate_tokens() <= self.context_budget:
                    return

        # 2. Drop whole earlier tasks (never the system prompt or the current task).
        while self.estimate_tokens() > self.context_budget:
            starts = [i for i, m in enumerate(self.messages) if isinstance(m, HumanMessage)]
            if len(starts) < 2:
                break
            del self.messages[starts[0]:starts[1]]

        # 3. Still too big (one huge task): cut every remaining tool output harder.
        if self.estimate_tokens() > self.context_budget:
            for i, message in enumerate(self.messages):
                if isinstance(message, ToolMessage) and len(_text_of(message.content)) > 1500:
                    self.messages[i] = ToolMessage(
                        content=_truncate(_text_of(message.content), 1500),
                        tool_call_id=message.tool_call_id,
                    )

    # ---- the loop ----------------------------------------------------------

    async def run(self, task: str) -> str:
        """Work on one task until the model stops asking for tools."""

        task_start = len(self.messages)
        self.messages.append(HumanMessage(task))
        self._seen_calls.clear()
        self._file_reads.clear()

        for _ in range(self.settings.max_steps):
            self._fit_context()
            response = await self._call_model()

            if response is None:
                # The call failed. Remove this task from the history so the next
                # task doesn't start by answering this one.
                del self.messages[task_start:]
                return ""

            self.messages.append(response)

            # No tool calls means the model has given its final answer.
            if not response.tool_calls:
                return _text_of(response.content)

            for call in response.tool_calls:
                await self._run_tool(call)

        # Close the task in the history, so the next task doesn't pick it up again.
        self.messages.append(
            AIMessage(content="(I stopped here because I reached the step limit before finishing.)")
        )
        self.events.on_error(
            f"Stopped after {self.settings.max_steps} steps without finishing. "
            "Give a more specific task or type 'continue'."
        )
        return ""

    async def _call_model(self) -> AIMessage | None:
        """Stream one model response. Returns None if the call failed."""

        # Groq sometimes rejects a badly formed tool call from the model
        # ("tool_use_failed"). Asking again usually works, so retry once.
        for attempt in range(2):
            self.events.on_thinking()
            full = None
            try:
                async for chunk in self.llm.astream(self.messages):
                    text = _text_of(chunk.content)
                    if text:
                        self.events.on_token(text)
                    full = chunk if full is None else full + chunk
            except Exception as exc:  # network errors, rate limits, bad requests
                message = str(exc)
                if attempt == 0 and "tool_use_failed" in message:
                    continue
                if "Request too large" in message or "413" in message:
                    message = (
                        "The request was bigger than the model's size limit. "
                        "Type /clear and try again, or lower CORVUS_CONTEXT_BUDGET_GROQ in .env.\n\n"
                        + message
                    )
                self.events.on_error(
                    f"Model call failed, so this task was stopped and removed from the "
                    f"conversation.\n{type(exc).__name__}: {message}"
                )
                return None

            if full is None:
                return AIMessage(content="")
            return message_chunk_to_message(full)

        return None

    async def _run_tool(self, call: dict[str, Any]) -> None:
        """Run one tool call and add its result to the conversation."""

        name = call["name"]
        args = dict(call.get("args") or {})
        server = name.split("__", 1)[0] if "__" in name else "local"

        # directory_tree on a project root walks into venv/.git and returns
        # tens of thousands of lines, so always exclude those folders.
        if name.endswith("__directory_tree"):
            excludes = set(args.get("excludePatterns") or []) | set(DEFAULT_TREE_EXCLUDES)
            args["excludePatterns"] = sorted(excludes)

        self.events.on_tool_start(name, server, args)

        mutating = self.registry.confirmation_gate.requires_confirmation(name)
        key = name + json.dumps(args, sort_keys=True, default=str)

        file_key = ""
        if name.endswith("__read_text_file"):
            file_key = str(args.get("path", "")).replace("\\", "/").lstrip("./").lower()

        if name not in self.tool_names:
            result: Any = {
                "ok": False,
                "error": f"Unknown tool '{name}'. Available tools: {', '.join(sorted(self.tool_names))}",
            }
        elif file_key and self._file_reads.get(file_key, 0) >= 2:
            # Models sometimes re-read the same file with ever bigger head/tail values.
            result = {
                "ok": False,
                "error": (
                    "You have already read this file twice in this task and its content is above. "
                    "Don't read it again: answer with what you have, or read a different file."
                ),
            }
        elif not mutating and key in self._seen_calls:
            # Same read with the same arguments, and nothing changed since.
            result = {
                "ok": False,
                "error": (
                    "You already made this exact call in this task and nothing has changed since. "
                    "Use the earlier result, look at a different part, or give your answer."
                ),
            }
        else:
            try:
                result = await self.registry.execute(name, args)
            except Exception as exc:  # never let one bad tool call end the task
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            if mutating:
                # Files or state may have changed, so earlier reads are stale.
                self._seen_calls.clear()
                self._file_reads.clear()
            else:
                self._seen_calls.add(key)
                self._call_keys[call["id"]] = key
                if file_key:
                    self._file_reads[file_key] = self._file_reads.get(file_key, 0) + 1

        text, ok = format_tool_result(result)
        self.events.on_tool_end(name, text, ok)

        if name.endswith("__read_text_file") and ok:
            content = _truncate_file_read(text, self.settings.max_file_read_chars, args)
        else:
            content = _truncate(text, self.settings.max_tool_result_chars)

        self.messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
