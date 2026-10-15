
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "chroma"

COLLECTION_NAME = "fastapi_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RAGRetriever:
    """Retrieve and rerank documentation chunks."""

    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(DB_DIR))

        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                "RAG index is missing. Run "
                "'python -m rag_server.ingest' first."
            ) from exc

        if self.collection.count() == 0:
            raise RuntimeError("The RAG index is empty.")

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def search(self, query: str, k: int = 5) -> list[dict]:
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10.")

        candidate_count = min(20, self.collection.count())

        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        matches = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = matches["documents"][0]
        metadatas = matches["metadatas"][0]
        distances = matches["distances"][0]

        if not documents:
            return []

        pairs = [[query, document] for document in documents]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(documents, metadatas, distances, scores),
            key=lambda item: float(item[3]),
            reverse=True,
        )

        results = []

        for document, metadata, distance, score in ranked[:k]:
            results.append({
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "vector_distance": float(distance),
                "rerank_score": float(score),
            })

        return results
