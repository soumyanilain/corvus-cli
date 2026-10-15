from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "fastapi_docs"
DB_DIR = ROOT / "data" / "chroma"

COLLECTION_NAME = "fastapi_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 64


def split_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main() -> None:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documentation directory does not exist: {DOCS_DIR}"
        )

    files = sorted(DOCS_DIR.rglob("*.md"))

    if not files:
        raise RuntimeError(
            "No Markdown files found in data/fastapi_docs."
        )

    documents = []
    metadatas = []
    ids = []

    for path in files:
        source = path.relative_to(DOCS_DIR).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")

        for index, chunk in enumerate(split_text(text)):
            chunk_id = hashlib.sha256(
                f"{source}:{index}".encode("utf-8")
            ).hexdigest()

            documents.append(chunk)
            metadatas.append({
                "source": source,
                "chunk_index": index,
            })
            ids.append(chunk_id)

    if not documents:
        raise RuntimeError("No non-empty document chunks were found.")

    print(f"Found {len(files)} Markdown files.")
    print(f"Prepared {len(documents)} chunks.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Build a temporary collection first.
    new_name = "fastapi_docs_new"

    try:
        client.delete_collection(new_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=new_name,
        metadata={"hnsw:space": "cosine"},
    )

    try:
        for start in range(0, len(documents), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_docs = documents[start:end]

            embeddings = model.encode(
                batch_docs,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            collection.add(
                ids=ids[start:end],
                documents=batch_docs,
                metadatas=metadatas[start:end],
                embeddings=embeddings,
            )

            print(
                f"Indexed {min(end, len(documents))}"
                f"/{len(documents)} chunks"
            )

        # Remove the old collection and rename the new one.
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection.modify(name=COLLECTION_NAME)

    except Exception:
        # Clean up the temporary collection if ingestion fails.
        try:
            client.delete_collection(new_name)
        except Exception:
            pass
        raise

    print(f"Successfully indexed {len(documents)} chunks.")
    print(f"Persistent database: {DB_DIR}")


if __name__ == "__main__":
    main()
