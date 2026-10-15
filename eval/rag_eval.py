"""
Evaluate the advanced RAG technique: vector search alone vs. vector search + reranking.

    python -m eval.rag_eval

For each question in eval/rag_questions.json we know which docs page(s) answer it.
We run the same retriever twice:
  - baseline: top 4 chunks by vector (cosine) similarity
  - reranked: top 20 by vector similarity, re-scored by the cross-encoder, keep top 4
and measure:
  - Hit@1: the first result comes from a correct page
  - Hit@4: at least one of the 4 results comes from a correct page
  - MRR:   1 / (rank of the first correct result), 0 if none; averaged
  - Recall@20: a correct page is somewhere in the 20 candidates
              (reranking can't find what vector search didn't fetch)

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
    print(f"Loading models and running {len(questions)} questions...")
    retriever.warm_up()

    rows = []
    base_ranks, rr_ranks, recall20 = [], [], []
    base_times, rr_times = [], []
    for q in questions:
        t = time.perf_counter()
        base = retriever.search(q["question"], k=config.CANDIDATES, rerank=False)
        base_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        reranked = retriever.search(q["question"], k=K, rerank=True)
        rr_times.append(time.perf_counter() - t)

        candidates = [r.source for r in base]           # all 20, in vector order
        b_rank = first_hit_rank(candidates[:K], q["expected"])
        r_rank = first_hit_rank([r.source for r in reranked], q["expected"])
        in20 = first_hit_rank(candidates, q["expected"]) is not None

        base_ranks.append(b_rank)
        rr_ranks.append(r_rank)
        recall20.append(in20)
        rows.append(
            {
                "question": q["question"],
                "expected": q["expected"],
                "baseline_rank": b_rank,
                "reranked_rank": r_rank,
                "in_top20": in20,
                "baseline_top": candidates[:K],
                "reranked_top": [r.source for r in reranked],
            }
        )
        mark = lambda r: str(r) if r else "-"
        print(f"  base {mark(b_rank):>2} | rerank {mark(r_rank):>2} | {q['question'][:70]}")

    summary = {
        "questions": len(questions),
        "k": K,
        "candidates": config.CANDIDATES,
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "baseline": summarize(base_ranks, K),
        "reranked": summarize(rr_ranks, K),
        "recall@20": round(sum(recall20) / len(recall20), 3),
        "avg_latency_ms": {
            "baseline": round(statistics.mean(base_times) * 1000, 1),
            "reranked": round(statistics.mean(rr_times) * 1000, 1),
        },
        "improved": sum(1 for b, r in zip(base_ranks, rr_ranks) if (r or 99) < (b or 99)),
        "worsened": sum(1 for b, r in zip(base_ranks, rr_ranks) if (r or 99) > (b or 99)),
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rag_eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    b, r = summary["baseline"], summary["reranked"]
    md = [
        "# RAG evaluation: vector search vs. vector search + reranking",
        "",
        f"{len(questions)} questions, top {K} results, reranking over {config.CANDIDATES} candidates.",
        f"Embedding model: `{config.EMBED_MODEL}`. Reranker: `{config.RERANK_MODEL}`.",
        "",
        f"| Metric | Vector search only | + Cross-encoder reranking |",
        f"|---|---|---|",
        f"| Hit@1 | {b['hit@1']:.0%} | {r['hit@1']:.0%} |",
        f"| Hit@{K} | {b[f'hit@{K}']:.0%} | {r[f'hit@{K}']:.0%} |",
        f"| MRR | {b['mrr']:.3f} | {r['mrr']:.3f} |",
        f"| Avg latency | {summary['avg_latency_ms']['baseline']} ms | {summary['avg_latency_ms']['reranked']} ms |",
        "",
        f"Recall@{config.CANDIDATES} (correct page anywhere in the candidates): {summary['recall@20']:.0%}",
        f"Reranking moved the first correct result up for {summary['improved']} questions "
        f"and down for {summary['worsened']}.",
        "",
        "| Question | Vector rank | Reranked rank |",
        "|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['question']} | {row['baseline_rank'] or '-'} | {row['reranked_rank'] or '-'} |"
        )
    (RESULTS / "rag_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    print(f"              Hit@1   Hit@{K}   MRR")
    print(f"  baseline    {b['hit@1']:.0%}    {b[f'hit@{K}']:.0%}    {b['mrr']:.3f}")
    print(f"  reranked    {r['hit@1']:.0%}    {r[f'hit@{K}']:.0%}    {r['mrr']:.3f}")
    print(f"  recall@{config.CANDIDATES}: {summary['recall@20']:.0%}   "
          f"improved {summary['improved']}, worsened {summary['worsened']}")
    print(f"\nSaved to {RESULTS / 'rag_eval.md'}")
    return summary


if __name__ == "__main__":
    main()
