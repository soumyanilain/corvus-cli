"""
docs-rag: our custom MCP server. It answers questions from the FastAPI
documentation using vector search + cross-encoder reranking.

Corvus starts it automatically (see servers.json). To run it by hand:

    python -m rag_server.server

It talks MCP over stdin/stdout, so nothing in this process may print to
stdout. Logs go to stderr.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import config
from .retriever import DocsRetriever, Result

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
log = logging.getLogger("docs-rag")

server = MCPServer(
    name="docs-rag",
    instructions=(
        "Searches the official FastAPI documentation. Use search_docs before "
        "writing or changing FastAPI code, or when a question is about FastAPI."
    ),
)
retriever = DocsRetriever()

NOT_BUILT = (
    "The FastAPI docs index has not been built yet. Ask the user to run "
    "`python -m rag_server.ingest` once, then restart Corvus."
)


def format_results(query: str, results: list[Result]) -> str:
    """Plain text the LLM can read: where each passage is from, then the passage."""

    if not results:
        return f"No documentation found for: {query}"

    lines = [
        f"Top {len(results)} FastAPI documentation passages for: {query}",
        f"(vector search over {config.CANDIDATES} candidates, reranked with a cross-encoder)",
        "",
    ]
    for r in results:
        where = f"{r.title} > {r.section}" if r.section else r.title
        score = f"relevance {r.rerank_score:.2f}" if r.rerank_score is not None else f"similarity {r.vector_score:.2f}"
        body = r.text.split("\n\n", 1)[-1]  # drop the "Title > Section" prefix, shown above
        if len(body) > config.MAX_CHARS_PER_RESULT:
            body = body[: config.MAX_CHARS_PER_RESULT] + " ..."
        lines += [f"[{r.rank}] {where}  ({score})", f"Source: {r.url}", body, ""]
    return "\n".join(lines).strip()


@server.tool(
    description=(
        "Search the official FastAPI documentation (tutorials, advanced guides, deployment). "
        "Use it for questions about FastAPI: path operations, request/response models, "
        "validation, dependencies, security, middleware, testing, and code examples. "
        "Returns the most relevant documentation passages with links."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def search_docs(query: str, k: int = config.TOP_K) -> str:
    """Search the FastAPI docs and return the k most relevant passages."""

    if not retriever.ready():
        return NOT_BUILT

    k = max(1, min(int(k), 8))
    # Model inference is CPU work; run it in a thread so the server stays responsive.
    results = await asyncio.to_thread(retriever.search, query, k)
    return format_results(query, results)


def _warm_up() -> None:
    """Load the vector store and both models in the background at startup."""

    if not retriever.ready():
        return
    try:
        retriever.warm_up()
    except Exception:  # the first search will report the real error
        log.exception("warm-up failed")


def main() -> None:
    threading.Thread(target=_warm_up, daemon=True).start()
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
