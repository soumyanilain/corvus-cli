# RAG evaluation: vector search vs. reranking vs. rank fusion

25 questions, top 4 results, re-ordering 20 candidates.
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. RRF k = 60.

| Metric | Vector search only | + Cross-encoder rerank | + Rerank with rank fusion |
|---|---|---|---|
| Hit@1 | 76% | 80% | 80% |
| Hit@4 | 92% | 88% | 92% |
| MRR | 0.830 | 0.830 | 0.850 |
| Avg latency | 18 ms | 835 ms | 882 ms |

Recall@20 (correct page anywhere in the candidates): 96%

| Question | Vector | Rerank | Fusion |
|---|---|---|---|
| How do I declare an optional query parameter that defaults to None? | 2 | 1 | 1 |
| How can I restrict a query string parameter to at most 50 characters? | 1 | 1 | 1 |
| How do I make sure an integer in the URL path is greater than zero? | - | 1 | 4 |
| How do I receive JSON in a POST request and validate it with a Pydantic model? | 4 | - | - |
| My React frontend on another port gets blocked by the browser when calling my API. How do I allow it? | 1 | 1 | 1 |
| How do I return a 404 error when an item doesn't exist? | 1 | 1 | 1 |
| How do I write automated tests for my endpoints with pytest? | 1 | 1 | 1 |
| How can users upload an image to my endpoint? | 1 | 1 | 1 |
| Where do I put code that should run once when the app starts and once when it shuts down? | 2 | 1 | 1 |
| How do I send an email after returning the response, without making the client wait? | 1 | 1 | 1 |
| How do I reuse the same pagination parameters across many endpoints? | - | - | - |
| How do I issue and verify JWT access tokens for logged-in users? | 1 | 1 | 1 |
| My main.py is getting huge. How do I split routes into several modules? | 1 | 1 | 1 |
| How do I make a create endpoint respond with status 201 instead of 200? | 1 | 4 | 2 |
| How do I make sure the user's password is never included in the API response? | 1 | 1 | 1 |
| How do I read the User-Agent sent by the client? | 1 | 1 | 1 |
| How do I serve CSS and image files from a folder? | 1 | - | 1 |
| How do I keep a two-way real-time connection open with the browser for a chat app? | 1 | 1 | 1 |
| How do I containerize my FastAPI app? | 1 | 1 | 2 |
| How do I store and query data in a SQL database from my endpoints? | 1 | 1 | 1 |
| How do I measure how long each request takes and add it as a response header? | 1 | 1 | 1 |
| How do I accept username and password sent as an HTML form instead of JSON? | 2 | 2 | 1 |
| Should my path operation functions be async def or normal def? | 1 | 1 | 1 |
| In tests, how do I replace a database dependency with a fake one? | 1 | 1 | 1 |
| How do I load configuration like API keys from environment variables? | 1 | 1 | 1 |
