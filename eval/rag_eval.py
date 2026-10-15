"""
Evaluate the advanced RAG technique: vector search alone vs. reranking vs. rank fusion.

    python -m eval.rag_eval

For each question in eval/rag_questions.json we know which docs page(s) answer it.
We run the same retriever three ways (see rag_server/retriever.py):
  - vector: top 4 chunks by vector (cosine) similarity              (baseline)
  - rerank: top 20 by vector similarity, re-ordered by the cross-encoder, keep 4
  - fusion: top 20, ordered by Reciprocal Rank Fusion of both rankings, keep 4
and measure:
  - Hit@1:     the first result comes from a correct page
  - Hit@4:     at least one of the 4 results comes from a correct page
  - MRR:       1 / (rank of the first correct result), 0 if none; averaged
  - Recall@20: a correct page is somewhere in the 20 candidates
               (no re-ordering can find what vector search didn't fetch)

Results are written to eval/results/rag_eval.json and eval/results/rag_eval.md.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from rag_server import config
from rag_server.retriever import DocsRetriever

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
K = config.TOP_K
METHODS = ["vector", "rerank", "fusion"]
LABELS = {"vector": "Vector search only", "rerank": "+ Cross-encoder rerank", "fusion": "+ Rerank with rank fusion"}


def first_hit_rank(sources: list[str], expected: list[str]) -> int | None:
    for rank, source in enumerate(sources, start=1):
        if source in expected:
            return rank
    return None


def summarize(ranks: list[int | None], k: int) -> dict[str, float]:
    n = len(ranks)
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        f"hit@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def main(retriever: DocsRetriever | None = None) -> dict:
    retriever = retriever or DocsRetriever()
    if not retriever.ready():
        sys.exit("Index not built. Run: python -m rag_server.ingest")

    questions = json.loads((HERE / "rag_questions.json").read_text(encoding="utf-8"))
    print(f"Loading models and running {len(questions)} questions x {len(METHODS)} methods...")
    retriever.warm_up()

    ranks = {m: [] for m in METHODS}
    times = {m: [] for m in METHODS}
    recall20 = []
    rows = []
    for q in questions:
        tops = {}
        for m in METHODS:
            t = time.perf_counter()
            # For "vector" fetch all 20 so we can also measure Recall@20.
            found = retriever.search(q["question"], k=config.CANDIDATES if m == "vector" else K, method=m)
            times[m].append(time.perf_counter() - t)
            tops[m] = [r.source for r in found]

        in20 = first_hit_rank(tops["vector"], q["expected"]) is not None
        tops["vector"] = tops["vector"][:K]
        row = {"question": q["question"], "expected": q["expected"], "in_top20": in20}
        for m in METHODS:
            rank = first_hit_rank(tops[m], q["expected"])
            ranks[m].append(rank)
            row[f"{m}_rank"] = rank
            row[f"{m}_top"] = tops[m]
        recall20.append(in20)
        rows.append(row)
        show = " | ".join(f"{m} {row[f'{m}_rank'] or '-':>2}" for m in METHODS)
        print(f"  {show} | {q['question'][:62]}")

    def changed(m: str, better: bool) -> int:
        out = 0
        for b, r in zip(ranks["vector"], ranks[m]):
            b, r = b or 99, r or 99
            out += (r < b) if better else (r > b)
        return out

    summary = {
        "questions": len(questions),
        "k": K,
        "candidates": config.CANDIDATES,
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "rrf_k": config.RRF_K,
        "methods": {m: summarize(ranks[m], K) for m in METHODS},
        "recall@20": round(sum(recall20) / len(recall20), 3),
        "avg_latency_ms": {m: round(statistics.mean(times[m]) * 1000, 1) for m in METHODS},
        "vs_vector": {m: {"improved": changed(m, True), "worsened": changed(m, False)} for m in METHODS[1:]},
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rag_eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    s = summary["methods"]
    md = [
        "# RAG evaluation: vector search vs. reranking vs. rank fusion",
        "",
        f"{len(questions)} questions, top {K} results, re-ordering {config.CANDIDATES} candidates.",
        f"Embedding model: `{config.EMBED_MODEL}`. Reranker: `{config.RERANK_MODEL}`. RRF k = {config.RRF_K}.",
        "",
        "| Metric | " + " | ".join(LABELS[m] for m in METHODS) + " |",
        "|---|" + "---|" * len(METHODS),
        "| Hit@1 | " + " | ".join(f"{s[m]['hit@1']:.0%}" for m in METHODS) + " |",
        f"| Hit@{K} | " + " | ".join(f"{s[m][f'hit@{K}']:.0%}" for m in METHODS) + " |",
        "| MRR | " + " | ".join(f"{s[m]['mrr']:.3f}" for m in METHODS) + " |",
        "| Avg latency | " + " | ".join(f"{summary['avg_latency_ms'][m]:.0f} ms" for m in METHODS) + " |",
        "",
        f"Recall@{config.CANDIDATES} (correct page anywhere in the candidates): {summary['recall@20']:.0%}",
        "",
        "| Question | Vector | Rerank | Fusion |",
        "|---|---|---|---|",
    ]
    for row in rows:
        md.append(f"| {row['question']} | " + " | ".join(str(row[f'{m}_rank'] or '-') for m in METHODS) + " |")
    (RESULTS / "rag_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    print(f"              Hit@1   Hit@{K}   MRR     latency")
    for m in METHODS:
        print(f"  {m:<10}  {s[m]['hit@1']:>5.0%}  {s[m][f'hit@{K}']:>5.0%}   {s[m]['mrr']:.3f}  "
              f"{summary['avg_latency_ms'][m]:>6.0f} ms")
    print(f"  recall@{config.CANDIDATES}: {summary['recall@20']:.0%}")
    for m in METHODS[1:]:
        v = summary["vs_vector"][m]
        print(f"  {m} vs vector: improved {v['improved']}, worsened {v['worsened']}")
    print(f"\nSaved to {RESULTS / 'rag_eval.md'}")
    return summary


if __name__ == "__main__":
    main()
