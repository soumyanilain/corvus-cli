"""
Provider layer: one function that returns a chat model for any backend.

The agent loop only ever calls get_chat_model(). It never imports Groq or
Ollama directly, so adding another provider (OpenAI, Anthropic, ...) means
adding one branch here and nothing else.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings

SUPPORTED_PROVIDERS = ("groq", "ollama")


class ProviderError(RuntimeError):
    """Raised when a provider can't be created (unknown name, missing key...)."""


def get_chat_model(
    provider: str,
    model: str | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """
    Return a LangChain chat model that supports .bind_tools() and .astream().

    provider: "groq" (cloud) or "ollama" (local)
    model:    model name; if None, the default from Settings / .env is used
    """

    settings = settings or Settings()
    provider = provider.lower().strip()
    model = model or settings.default_model(provider)

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise ProviderError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(free key at console.groq.com)."
            )

        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=0,
            # The free tier has a tokens-per-minute limit. The Groq client
            # waits and retries on a 429, so a few retries keep long tasks
            # going instead of crashing.
            max_retries=6,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=settings.ollama_num_ctx,
        )

    raise ProviderError(
        f"Unknown provider '{provider}'. "
        f"Use one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )
