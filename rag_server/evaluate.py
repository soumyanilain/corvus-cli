
from rag_server.retriever import RAGRetriever


# Replace these expected filenames if the documentation version
# you downloaded uses different paths.
TEST_CASES = [
    {
        "query": "How do I define a POST endpoint in FastAPI?",
        "expected_source": "tutorial",
    },
    {
        "query": "How does FastAPI validate request body data?",
        "expected_source": "tutorial",
    },
    {
        "query": "How do I inject dependencies into a FastAPI route?",
        "expected_source": "tutorial",
    },
    {
        "query": "How do I define a response model?",
        "expected_source": "tutorial",
    },
]


def source_matches(source: str, expected: str) -> bool:
    return expected.lower() in source.lower()


def main() -> None:
    rag = RAGRetriever()

    vector_hits = 0
    reranked_hits = 0
    vector_reciprocal_ranks = []
    reranked_reciprocal_ranks = []

    for case in TEST_CASES:
        query = case["query"]
        expected = case["expected_source"]

        # Retrieve a wider candidate set to measure baseline ranking.
        results = rag.search(query, k=10)

        # These results are already reranked. For a proper baseline,
        # query ChromaDB directly in a separate vector-only function.
        # Do not label this list as the original vector ranking.
        found_rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if source_matches(result["source"], expected)
            ),
            None,
        )

        if found_rank:
            reranked_hits += 1
            reranked_reciprocal_ranks.append(1 / found_rank)
        else:
            reranked_reciprocal_ranks.append(0.0)

        print(f"\nQuestion: {query}")
        print("Top result:", results[0]["source"] if results else "None")

    print("\nReranked retrieval results")
    print(f"Questions: {len(TEST_CASES)}")
    print(f"Hit rate @10: {reranked_hits / len(TEST_CASES):.2f}")
    print(
        "MRR: "
        f"{sum(reranked_reciprocal_ranks) / len(TEST_CASES):.3f}"
    )
    print(
        "\nNote: These initial tests use broad expected-source labels. "
        "Use exact expected document paths and implement vector-only "
        "ranking before claiming a baseline-versus-reranking improvement."
    )


if __name__ == "__main__":
    main()
