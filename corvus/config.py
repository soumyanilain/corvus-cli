"""
Settings for Corvus.

Everything can be changed from the .env file in the project root,
so nobody has to edit code to switch models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Finds the .env file in the project root.
load_dotenv()

# Project root (the folder that contains corvus/, mcp_tools/, servers.json).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tools we don't show to the model. They are duplicates or not useful for
# coding tasks. Fewer tools means smaller prompts, which matters on Groq's
# free tier (limited tokens per minute).
HIDDEN_TOOLS = {
    "filesystem__read_file",                  # deprecated alias of read_text_file
    "filesystem__read_media_file",            # images/audio, not needed for code
    "filesystem__list_directory_with_sizes",  # list_directory is enough
    "filesystem__read_multiple_files",        # one file at a time keeps outputs small
    "filesystem__get_file_info",              # rarely needed for coding tasks
    "filesystem__list_allowed_directories",   # the system prompt already says the workspace
    "tavily__tavily_crawl",                   # search + extract are enough
    "tavily__tavily_map",
    "tavily__tavily_research",                # slow and heavily rate limited
}

# Folders directory_tree always skips (otherwise one call on the project
# root returns tens of thousands of lines from venv/.git).
DEFAULT_TREE_EXCLUDES = ["venv", ".venv", ".git", "node_modules", "__pycache__"]


@dataclass
class Settings:
    """Runtime settings. Defaults come from environment variables."""

    provider: str = os.getenv("CORVUS_PROVIDER", "groq")
    groq_model: str = os.getenv("CORVUS_GROQ_MODEL", "openai/gpt-oss-120b")
    ollama_model: str = os.getenv("CORVUS_OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Safety cap on reason -> act -> observe cycles for one task. Kept low so a
    # model stuck in a loop can't burn through the provider's daily token limit.
    max_steps: int = int(os.getenv("CORVUS_MAX_STEPS", "15"))

    # Tool results longer than this are cut before going back to the model,
    # so one big file or command output can't blow up the context window.
    max_tool_result_chars: int = int(os.getenv("CORVUS_MAX_TOOL_RESULT_CHARS", "6000"))

    # File reads get a bigger limit, so a normal source file arrives in one read.
    max_file_read_chars: int = int(os.getenv("CORVUS_MAX_FILE_READ_CHARS", "12000"))

    # Max size (in tokens, roughly) of one request to the model, including the
    # tool definitions. Groq's free tier rejects any request over 8,000 tokens,
    # so we stay well under that. Older tool outputs get trimmed to fit.
    context_budget_groq: int = int(os.getenv("CORVUS_CONTEXT_BUDGET_GROQ", "6000"))
    context_budget_ollama: int = int(os.getenv("CORVUS_CONTEXT_BUDGET_OLLAMA", "12000"))

    # Ollama's default context window is small; tool definitions alone need ~4k tokens.
    ollama_num_ctx: int = int(os.getenv("CORVUS_OLLAMA_NUM_CTX", "16384"))

    def default_model(self, provider: str) -> str:
        return self.groq_model if provider == "groq" else self.ollama_model

    def context_budget(self, provider: str) -> int:
        return self.context_budget_groq if provider == "groq" else self.context_budget_ollama
