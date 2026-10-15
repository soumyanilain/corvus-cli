"""
One-time ingestion: FastAPI docs -> chunks -> embeddings -> ChromaDB.

    python -m rag_server.ingest            # build the index (skips if it already exists)
    python -m rag_server.ingest --force    # rebuild from scratch

Steps:
  1. Download the FastAPI docs (Markdown) and their code examples with a
     small git sparse checkout into data/fastapi-src.
  2. Clean each page: inline the code examples the docs pull in with
     {* ../../docs_src/... *}, strip HTML and MkDocs-only syntax.
  3. Split by Markdown headings, then into ~1000-character chunks. Each chunk
     starts with its page title and section, so it still makes sense alone.
  4. Embed every chunk with a sentence-transformers model.
  5. Store chunks + embeddings + metadata in a persistent ChromaDB collection.

After this runs once, every Corvus session reuses data/chroma.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from . import config

# A function that turns a list of texts into a list of vectors.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, str]


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_docs() -> None:
    """Sparse-clone only the docs and code examples from the FastAPI repo (~20 MB)."""

    if config.DOCS_DIR.exists() and config.CODE_DIR.exists():
        print(f"Docs already downloaded: {config.DOCS_DIR}")
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading FastAPI docs (git sparse checkout)...")
    if not config.DOCS_CHECKOUT.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             config.DOCS_REPO_URL, str(config.DOCS_CHECKOUT)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(config.DOCS_CHECKOUT), "sparse-checkout", "set",
         "docs/en/docs", "docs_src"],
        check=True,
    )


# ---------------------------------------------------------------------------
# 2. Load and clean
# ---------------------------------------------------------------------------

# {* ../../docs_src/first_steps/tutorial001_py310.py hl[3] ln[1:10] *}
INCLUDE_RE = re.compile(r"\{\*\s*(?P<path>[^\s*]+)(?P<opts>[^*]*)\*\}")
LINE_RANGE_RE = re.compile(r"ln\[(\d+):(\d+)\]")
HEADING_ANCHOR_RE = re.compile(r"[ \t]*\{\s*#[\w\-]+\s*\}[ \t]*$", re.MULTILINE)
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]*>")
ADMONITION_RE = re.compile(r"^[ \t]*///[ \t]*(\w+)?.*$", re.MULTILINE)


def _inline_code(match: re.Match[str], code_dir: Path) -> str:
    """Replace a {* ... *} include with the actual code, as a fenced block."""

    rel = match.group("path")
    if not rel.startswith("../../docs_src/"):
        return ""  # a few pages include FastAPI's own source code; skip those
    path = code_dir / rel.removeprefix("../../docs_src/")
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    line_range = LINE_RANGE_RE.search(match.group("opts"))
    if line_range:
        start, end = int(line_range.group(1)), int(line_range.group(2))
        lines = lines[start - 1:end]

    if len(lines) > config.MAX_INLINE_CODE_LINES:
        lines = lines[: config.MAX_INLINE_CODE_LINES] + ["# ... (example shortened)"]

    return "```python\n" + "\n".join(lines) + "\n```"


def clean_markdown(text: str, code_dir: Path) -> str:
    """Turn MkDocs-flavoured Markdown into plain Markdown with real code."""

    text = INCLUDE_RE.sub(lambda m: _inline_code(m, code_dir), text)
    text = HEADING_ANCHOR_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    # "/// tip" ... "///" boxes -> "Tip:" ... (closing marker removed)
    text = ADMONITION_RE.sub(lambda m: f"{m.group(1).capitalize()}:" if m.group(1) else "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_url(rel_path: str) -> str:
    """tutorial/first-steps.md -> https://fastapi.tiangolo.com/tutorial/first-steps/"""

    path = rel_path.removesuffix(".md")
    if path == "index":
        return config.DOCS_SITE + "/"
    if path.endswith("/index"):
        path = path[: -len("/index")]
    return f"{config.DOCS_SITE}/{path}/"


def load_pages(docs_dir: Path, code_dir: Path) -> list[tuple[str, str]]:
    """Return (relative path, cleaned text) for every page worth indexing."""

    pages = []
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir).as_posix()
        if path.name in config.SKIP_FILES or rel.split("/")[0] in config.SKIP_DIRS:
            continue
        text = clean_markdown(path.read_text(encoding="utf-8", errors="replace"), code_dir)
        if len(text) > 200:  # skip near-empty pages
            pages.append((rel, text))
    return pages


# ---------------------------------------------------------------------------
# 3. Chunk
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def split_by_headings(text: str) -> list[tuple[dict[int, str], str]]:
    """
    Split Markdown into sections at #, ## and ### headings.

    Written by hand instead of using LangChain's MarkdownHeaderTextSplitter,
    because that one strips leading spaces from every line, which destroys
    the indentation of Python code examples. Lines inside ``` fences are never
    treated as headings (Python comments also start with #).
    """

    sections: list[tuple[dict[int, str], str]] = []
    headings: dict[int, str] = {}
    lines: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((dict(headings), body))

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING_RE.match(line)
        if match:
            flush()
            lines = []
            level = len(match.group(1))
            headings = {lvl: h for lvl, h in headings.items() if lvl < level}
            headings[level] = match.group(2)
        else:
            lines.append(line)
    flush()
    return sections


def chunk_pages(pages: list[tuple[str, str]]) -> list[Chunk]:
    """Split each page by headings, then by size; prefix each chunk with where it's from."""

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    by_size = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks: list[Chunk] = []
    for rel, text in pages:
        sections = split_by_headings(text)
        page_title = next((h[1] for h, _ in sections if 1 in h), rel)
        n = 0
        for headings, body in sections:
            heading = " > ".join(headings[lvl] for lvl in (2, 3) if lvl in headings)
            for piece in by_size.split_text(body):
                if len(piece.strip()) < 40:
                    continue
                location = f"{page_title} > {heading}" if heading else page_title
                chunks.append(
                    Chunk(
                        id=f"{rel}::{n}",
                        text=f"{location}\n\n{piece}",
                        metadata={
                            "source": rel,
                            "title": page_title,
                            "section": heading,
                            "url": page_url(rel),
                        },
                    )
                )
                n += 1
    return chunks


# ---------------------------------------------------------------------------
# 4 + 5. Embed and store
# ---------------------------------------------------------------------------

def sentence_transformer_embedder() -> EmbedFn:
    """The real embedding model (downloads ~90 MB the first time)."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors = model.encode(
            list(texts),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vectors.tolist()

    return embed


def open_client(chroma_dir: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def index_exists(chroma_dir: Path = config.CHROMA_DIR) -> bool:
    return (chroma_dir / "ingest_info.json").exists()


def build_index(
    chunks: list[Chunk],
    embed: EmbedFn,
    chroma_dir: Path = config.CHROMA_DIR,
) -> int:
    """Embed all chunks and write them to a fresh ChromaDB collection."""

    client = open_client(chroma_dir)
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass  # didn't exist yet
    collection = client.create_collection(
        name=config.COLLECTION,
        metadata={"hnsw:space": "cosine"},  # we store normalized vectors; cosine distance
        embedding_function=None,            # we always pass our own embeddings
    )

    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL}...")
    vectors = embed([c.text for c in chunks])

    batch = 500
    for start in range(0, len(chunks), batch):
        part = chunks[start:start + batch]
        collection.add(
            ids=[c.id for c in part],
            documents=[c.text for c in part],
            metadatas=[c.metadata for c in part],
            embeddings=vectors[start:start + batch],
        )
    return collection.count()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the FastAPI docs vector index (run once).")
    parser.add_argument("--force", action="store_true", help="rebuild even if the index exists")
    args = parser.parse_args(argv)

    if index_exists() and not args.force:
        info = json.loads(config.INGEST_INFO.read_text(encoding="utf-8"))
        print(f"Index already built ({info['chunks']} chunks, {info['built_at']}). "
              "Nothing to do. Use --force to rebuild.")
        return

    started = time.time()
    download_docs()

    pages = load_pages(config.DOCS_DIR, config.CODE_DIR)
    print(f"Loaded {len(pages)} pages.")

    chunks = chunk_pages(pages)
    print(f"Split into {len(chunks)} chunks.")

    count = build_index(chunks, sentence_transformer_embedder())

    config.INGEST_INFO.write_text(
        json.dumps(
            {
                "chunks": count,
                "pages": len(pages),
                "embed_model": config.EMBED_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "built_at": time.strftime("%Y-%m-%d %H:%M"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done: {count} chunks stored in {config.CHROMA_DIR} "
          f"in {time.time() - started:.0f}s.")


if __name__ == "__main__":
    sys.exit(main())
