
from mcp.server.fastmcp import FastMCP

from rag_server.retriever import RAGRetriever


mcp = FastMCP("corvus-rag-server")
_retriever = None


def get_retriever() -> RAGRetriever:
    global _retriever

    if _retriever is None:
        _retriever = RAGRetriever()

    return _retriever


@mcp.tool()
def search_docs(query: str, k: int = 5) -> dict:
    """
    Search the indexed FastAPI documentation.

    Use when a coding task needs documentation about routes,
    request validation, dependencies, response models, or APIs.

    Args:
        query: A specific question about FastAPI.
        k: Number of reranked passages to return, from 1 to 10.
    """
    try:
        results = get_retriever().search(query, k)

        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    except (ValueError, RuntimeError) as exc:
        return {
            "query": query,
            "result_count": 0,
            "results": [],
            "error": str(exc),
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
