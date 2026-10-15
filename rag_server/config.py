"""
Settings for the custom RAG MCP server (docs-rag).

All paths are absolute and based on this file's location, so the server
works no matter which folder Corvus is started from.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Where the FastAPI docs are downloaded to (git sparse checkout).
DOCS_REPO_URL = "https://github.com/fastapi/fastapi.git"
DOCS_CHECKOUT = DATA_DIR / "fastapi-src"
DOCS_DIR = DOCS_CHECKOUT / "docs" / "en" / "docs"
CODE_DIR = DOCS_CHECKOUT / "docs_src"

# Persistent vector database. Built once by `python -m rag_server.ingest`.
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "fastapi_docs"
INGEST_INFO = CHROMA_DIR / "ingest_info.json"

# Models (downloaded from Hugging Face the first time, then cached).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Chunking.
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks
MAX_INLINE_CODE_LINES = 60  # longer code examples are cut when inlined

# Retrieval.
CANDIDATES = 20        # vector search returns this many...
TOP_K = 4              # ...and reranking keeps the best TOP_K
# "vector", "rerank" or "fusion" (see retriever.py). Chosen from eval/rag_eval.py results.
RETRIEVAL_METHOD = os.getenv("RAG_METHOD", "fusion")
RRF_K = 60             # standard Reciprocal Rank Fusion constant
MAX_CHARS_PER_RESULT = 1100  # keeps the tool's answer small enough for Groq's free tier

DOCS_SITE = "https://fastapi.tiangolo.com"

# Pages that would only add noise: a 700 KB changelog, auto-generated API
# stubs, and community/meta pages with no technical content.
SKIP_FILES = {
    "release-notes.md",
    "_llm-test.md",
    "translation-banner.md",
    "translations.md",
    "contributing.md",
    "management.md",
    "management-tasks.md",
    "fastapi-people.md",
    "external-links.md",
    "newsletter.md",
    "help-fastapi.md",
}
SKIP_DIRS = {"reference", "css", "js", "img"}
