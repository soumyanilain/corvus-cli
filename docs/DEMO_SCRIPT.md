# Demo video script (about 6 minutes)

The video must show two non-trivial tasks done on their own, with **all three MCP servers visibly called** (filesystem, Tavily, docs-rag) and the tool calls readable on screen.

## Before recording

```powershell
cd D:\corvus-cli
venv\Scripts\Activate.ps1
Remove-Item -Recurse -Force D:\corvus-demo -ErrorAction SilentlyContinue
Copy-Item -Recurse eval\sample_project D:\corvus-demo
```

- Close every code tab in VS Code; maximize the terminal panel; press **Ctrl +** 3–4 times so text is readable.
- Record with **Win + Alt + R** (Xbox Game Bar), OBS, or a Zoom meeting with only you in it.
- Each person can narrate the part they built (MCP client: Sogol, RAG server: Meghana, CLI: Sumiran).

## 1. Intro (30 s)

Show `docs/diagrams/architecture-v2.png`.

> "This is Corvus, our command-line coding agent. It runs an agentic loop on Groq or Ollama and gets its tools from three MCP servers: the official filesystem server, Tavily for web search, and docs-rag, our own server that searches the FastAPI docs with reranking."

## 2. Startup (30 s)

```powershell
python -m corvus --workspace D:\corvus-demo
```

Point at the `tools` line: `docs-rag (1), filesystem (8), local (2), tavily (2)`. Type `/tools`.

> "Nothing is hard-coded. The MCP client reads servers.json and lists each server's tools at startup. Tools marked 'asks first' need approval in confirm mode."

## 3. Task 1: fix and extend a FastAPI app (confirm mode, 2–3 min)

> "D:\corvus-demo is a small FastAPI app with a bug: asking for a book that doesn't exist crashes with a 500 error."

Type:

```
This folder is a small FastAPI app (app.py) with tests in test_app.py. First use the FastAPI docs search tool to look up how to return a 404 with HTTPException and how to validate a request body with a Pydantic model. Then add a POST /books endpoint that validates the body (title non-empty, author, year between 1450 and 2100) and returns the new book with status 201, and make GET /books/{book_id} return 404 instead of crashing when the id doesn't exist. Add tests for both, then run the tests with python -m pytest -q and fix anything that fails.
```

Narrate as panels appear:
- **search_docs via docs-rag**: "It looks up the FastAPI docs first: our RAG server, vector search plus cross-encoder reranking and rank fusion." (We ask for the docs search explicitly: in our LLM comparison, gpt-oss-120b skipped it when it already knew the answer.)
- **read_text_file via filesystem**: "It reads the code before changing it."
- **write_file / edit_file via filesystem** (yellow): "Changing a file is risky, so in confirm mode it asks." Type `y`.
- **run_command via local** (yellow): type `y`. "It runs the tests and reads the result. If something failed, it would fix it and run them again."

## 4. Task 2: web research and new files (auto mode, 1–2 min)

Type `/mode auto`:

> "Auto mode: it runs everything without asking."

Type:

```
Search the web for the latest stable versions of FastAPI and Pydantic. Then create requirements.txt in this folder pinning those two versions plus pytest and httpx, and write NOTES.md with a short summary of what's new in the latest FastAPI release and the link you got it from.
```

- **tavily_search / tavily_extract via tavily**: "Our external MCP server, Tavily, for live web search."
- **write_file via filesystem** (runs without a prompt now).

## 5. Wrap-up (30 s)

```
/exit
type D:\corvus-demo\NOTES.md
```

> "Every tool call is visible, the user stays in control with confirm mode, and the model can be switched with /model, for example to a local Ollama model."

If Groq says the **daily** token allowance is used up, stop recording, put a teammate's Groq key in `.env` (every account gets its own daily allowance) and start again. Don't switch to gpt-oss-20b: in our LLM comparison it couldn't make file edits. A short yellow `…` note (a retry or a one-minute wait) is normal; keep recording.
