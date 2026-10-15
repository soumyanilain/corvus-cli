"""
Retrieval with reranking (advanced RAG technique: NirDiamant RAG_Techniques #17).

    1. Embed the question and pull the CANDIDATES (20) closest chunks from ChromaDB.
       Embedding search is fast but rough: it finds text that *looks* similar.
    2. Score every candidate with a cross-encoder, which reads the question and
       the chunk *together* and judges how well the chunk answers it.
    3. Keep the TOP_K best by cross-encoder score.

rerank=False skips step 2 and returns the top results by vector similarity,
which is the baseline we compare against in eval/rag_eval.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from . import config
from .ingest import EmbedFn, index_exists, open_client

# Scores (question, passage) pairs; higher = more relevant.
RerankFn = Callable[[str, Sequence[str]], list[float]]


@dataclass
class Result:
    rank: int
    text: str
    source: str
    title: str
    section: str
    url: str
    vector_score: float          # cosine similarity from the vector search
    rerank_score: float | None   # cross-encoder score (None when rerank=False)


def cross_encoder_reranker() -> RerankFn:
    """The real reranking model (downloads ~90 MB the first time)."""

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(config.RERANK_MODEL)

    def rerank(query: str, passages: Sequence[str]) -> list[float]:
        scores = model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]

    return rerank


class DocsRetriever:
    """Loads models and the vector store lazily, once, and answers searches."""

    def __init__(
        self,
        chroma_dir: Path = config.CHROMA_DIR,
        embed: EmbedFn | None = None,
        rerank: RerankFn | None = None,
    ) -> None:
        self.chroma_dir = chroma_dir
        self._embed = embed
        self._rerank = rerank
        self._collection = None
        self._lock = threading.Lock()  # warm_up() may run in a background thread

    def ready(self) -> bool:
        """True once `python -m rag_server.ingest` has built the index."""

        return index_exists(self.chroma_dir)

    def warm_up(self) -> None:
        """Load everything now, so the first search isn't slow."""

        with self._lock:
            if self._collection is None:
                self._collection = open_client(self.chroma_dir).get_collection(config.COLLECTION)
            if self._embed is None:
                from .ingest import sentence_transformer_embedder

                self._embed = sentence_transformer_embedder()
            if self._rerank is None:
                self._rerank = cross_encoder_reranker()

    def search(
        self,
        query: str,
        k: int = config.TOP_K,
        rerank: bool = True,
        candidates: int = config.CANDIDATES,
    ) -> list[Result]:
        self.warm_up()

        query_vector = self._embed([query])[0]
        found = self._collection.query(
            query_embeddings=[query_vector],
            n_results=max(candidates, k),
            include=["documents", "metadatas", "distances"],
        )
        docs = found["documents"][0]
        metas = found["metadatas"][0]
        # Chroma returns cosine *distance*; similarity = 1 - distance.
        sims = [1.0 - d for d in found["distances"][0]]

        order = list(range(len(docs)))  # already sorted by vector similarity
        rerank_scores: list[float | None] = [None] * len(docs)
        if rerank and docs:
            scores = self._rerank(query, docs)
            rerank_scores = list(scores)
            order.sort(key=lambda i: scores[i], reverse=True)

        results = []
        for rank, i in enumerate(order[:k], start=1):
            meta = metas[i] or {}
            results.append(
                Result(
                    rank=rank,
                    text=docs[i],
                    source=meta.get("source", ""),
                    title=meta.get("title", ""),
                    section=meta.get("section", ""),
                    url=meta.get("url", ""),
                    vector_score=round(sims[i], 4),
                    rerank_score=None if rerank_scores[i] is None else round(rerank_scores[i], 4),
                )
            )
        return results
