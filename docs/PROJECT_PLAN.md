# Corvus — Project Plan

> Status: **planning draft, Oct 1 2026.** Written before implementation. We'll update the checkboxes as we go, but the plan itself stays as the record of what we intended.

**Course:** ITCS 5010, Group Project 2 (CLI Coding Assistant)
**Group:** Group __ <!-- fill in -->
**Key dates:** Initial plan Oct 1 · Check-in Oct 8 · Final submission Oct 15 (11:59 pm)

## Team and roles

Each person owns one area end to end (code, comments, and their part of the report). Everyone reviews someone else's PR at least once.

| Member | Role | Owns |
|---|---|---|
| Soumyanil Ain | **Lead / Agent core** | Repo and main branch, agentic loop, provider abstraction (Ollama + Groq), integration of all parts, LLM comparison, final report assembly |
| _Teammate 2_ | **MCP + Tools** | MCP client wrapper, `servers.json`, filesystem + Tavily servers, tool registry, confirmation gate, `run_command`, `grep_code` |
| _Teammate 3_ | **RAG server** | Doc ingestion, chunking, embeddings, ChromaDB, reranking, FastMCP server, RAG evaluation (with vs without reranking) |
| _Teammate 4_ | **CLI + Diagrams** | Rich REPL, streaming display, tool-call panels, slash commands, state + sequence diagrams, README setup section, video recording |

If we only have three people, Teammate 4's work splits: the CLI goes to the Lead, the diagrams and README go to Teammate 2.

## Interfaces we agree on first (by Sat Oct 3)

These let everyone build in parallel and plug together later without rewrites. Any change to these needs a message in the group chat first.

```python
# providers/  (Lead)
def get_chat_model(provider: str, model: str) -> BaseChatModel: ...
    # provider in {"ollama", "groq"}; returned model supports .bind_tools() and .astream()

# tools/  (MCP + Tools)
class ToolRegistry:
    async def load(self) -> list[BaseTool]: ...         # MCP tools + local tools
    def is_mutating(self, tool_name: str) -> bool: ...  # unknown tools count as mutating
    async def run(self, tool_call: dict) -> str: ...    # always returns a string, errors included

# rag_server/  (RAG)
@mcp.tool()
def search_docs(query: str, k: int = 5) -> str: ...     # top-k chunks with source file + heading

# agent/ → cli/  (Lead ↔ CLI)  the loop calls these; the CLI implements them
class AgentEvents(Protocol):
    def on_token(self, text: str) -> None: ...
    def on_tool_start(self, name: str, server: str, args: dict) -> None: ...
    def on_tool_end(self, name: str, preview: str, ok: bool) -> None: ...
    async def confirm(self, name: str, args: dict) -> Literal["y", "n", "a"]: ...
```

Because the loop only talks to the CLI through `AgentEvents`, the CLI can be built and tested with a fake loop, and the loop can be tested with a plain print-based version of the events.

## Timeline

### Week 0 — Planning (Thu Oct 1)
- [x] Pick the assistant name (Corvus) and overall design
- [x] Initial architecture diagram → `docs/ARCHITECTURE.md`
- [x] This plan → `docs/PROJECT_PLAN.md`
- [ ] Repo created, teammates and TAs added, planning commit pushed and tagged `planning-v1`
- [ ] Canvas submission (repo link)

### Week 1 — Foundations + two MCP servers (Fri Oct 2 → Thu Oct 8)

| Day | Goal | Who |
|---|---|---|
| Fri Oct 2 | Everyone finishes setup (Python 3.11+, Node 18+, Ollama + a model pulled, free Groq key, free Tavily key). Lead pushes the empty package skeleton, `requirements.txt`, `config.yaml`, `.env.example`. | All / Lead |
| Sat Oct 3 | Interfaces above agreed and committed as stubs. RAG: download FastAPI docs, decide chunk size. | All / RAG |
| Sun Oct 4 | Provider factory working for Ollama and Groq; test 2–3 Ollama models on tool calling and pick one. MCP client connects to the filesystem server and lists its tools. Ingestion script stores chunks in Chroma. **Team sync call.** | Lead / MCP / RAG |
| Mon Oct 5 | First version of the agent loop: model calls a filesystem tool, gets the result, answers. Tavily added to `servers.json`. Basic REPL with streaming. | Lead / MCP / CLI |
| Tue Oct 6 | Tool registry with confirm/auto modes. `run_command` + `grep_code`. RAG server exposes `search_docs` (plain vector search, no reranking yet). Tool-call panels in the CLI. | MCP / RAG / CLI |
| Wed Oct 7 | Integrate everything on `main`. RAG server added to `servers.json`. **Check-in dry run on a call**, write down challenges and next steps. | All |
| **Thu Oct 8** | **Check-in.** Target: filesystem + Tavily demonstrated end to end through the agent (rubric needs 2 of 3), RAG server shown too if stable. | All |

**Check-in definition of done:**
- `python -m corvus` starts, loads tools from at least two MCP servers, and shows them with `/tools`
- One task that reads a file, edits it (with a confirmation prompt), and answers
- One task that uses Tavily web search
- 3–4 talking points on challenges so far and the plan for week 2

### Week 2 — Advanced RAG, polish, evaluation, deliverables (Fri Oct 9 → Thu Oct 15)

| Day | Goal | Who |
|---|---|---|
| Fri Oct 9 | Fold in check-in feedback. Add cross-encoder reranking to the RAG server, with a flag to turn it off. | Lead / RAG |
| Sat Oct 10 | CLI polish: status spinners, diff preview in confirmations, `/model` and `/mode` switching, error display. Write the RAG eval set (~20 questions with expected source files). | CLI / RAG |
| Sun Oct 11 | Pick 2–3 non-trivial demo tasks and run them end to end. RAG eval: with vs without reranking (hit rate @5, MRR). LLM comparison: same task on Groq vs Ollama. Draft state diagram + sequence diagrams. | All |
| Mon Oct 12 | **Feature freeze.** Bug fixes only after today. Code comments pass. README setup instructions tested on a fresh clone by someone who didn't write them. | All |
| Tue Oct 13 | Record the demo video (two non-trivial tasks, all three MCP servers visibly called). Final diagrams. Each person sends their report section to the Lead. | CLI / All |
| Wed Oct 14 | Lead assembles the PDF report. Everyone reviews it. If the architecture changed, add `ARCHITECTURE_v2.md` and the updated diagram. | Lead / All |
| **Thu Oct 15** | Buffer. Final push, tag `v1.0`, submit repo link + video + PDF on Canvas. | Lead |

## Final deliverables checklist

| Deliverable | Rubric item | Owner |
|---|---|---|
| Agentic loop + provider abstraction (Ollama + Groq) | Agentic Loop & Core Architecture (20) | Lead |
| Tool calls visible, confirm + auto modes, streaming REPL | Tool Calling & CLI Interface (20) | CLI + MCP |
| Filesystem, Tavily, and custom RAG servers loaded dynamically | MCP Integration (20) | MCP |
| RAG server: ingest once, Chroma, reranking | Custom RAG MCP Server (20) | RAG |
| README, `requirements.txt`, comments, planning-before-coding history | Planning & Documentation (10) | All, Lead checks |
| PDF report: design decisions, LLM comparison, RAG analysis, reflection, original + updated diagrams | Written Reflection (10, shared) | Lead assembles |
| Demo video: two non-trivial tasks, all three servers visible | Video Demonstration (shared) | CLI |
| State diagram + sequence diagrams (≥2 scenarios, ≥3 operations total) | Architecture/workflow diagrams | CLI |

## Demo task ideas (to finalize Oct 11)
1. **Build from scratch:** "Create a small FastAPI todo API in `demo_app/` with Pydantic validation and pytest tests, then run the tests and fix anything that fails." Uses docs-rag, filesystem writes, `run_command`.
2. **Debug existing code:** "The tests in `sample_project/` are failing. Find out why, fix it, and rerun them." Uses `grep_code`, filesystem reads/edits, `run_command`, and Tavily for an unfamiliar error.
3. **Research + change:** "Find what changed in the latest release of library X and update our code to match." Uses Tavily, then filesystem edits.

The LLM comparison runs the same task (probably #1) on the Groq model and the Ollama model and records: did it finish, number of steps and tool calls, invalid tool calls, time taken, and how often we had to step in.

## How we work

**Git**
- `main` should always run. Work on branches named `feat/<area>-<thing>` (e.g. `feat/rag-rerank`) and open a PR. Someone other than the author reviews it before merging.
- Small commits with messages that say what changed. Everyone commits their own work so the history shows who did what.
- Planning docs were committed first and tagged `planning-v1`, before any implementation.
- **API keys never go in the repo.** Real keys live only in `.env` (gitignored). `.env.example` has the variable names with empty values.

**Communication**
- Group chat for day-to-day. Every other day, each person posts a 3-line update: done / next / blocked.
- Calls: Sun Oct 4 (integration sync), Wed Oct 7 (check-in dry run), Mon Oct 12 (feature freeze), Wed Oct 14 (report review).
- If you're blocked for more than half a day, say so in the chat. Don't wait for the next call.

**Tracking**
- Each row in the timeline becomes a GitHub Issue assigned to its owner, closed by the PR that finishes it.

## Risks

| Risk | What we'll do |
|---|---|
| Parts don't fit together at the end | Interfaces agreed by Oct 3; integrate on `main` daily from Oct 5, not on the last day. |
| Small Ollama models fail at tool calling | Test several on Oct 4; this becomes material for the LLM comparison either way. |
| One person falls behind | Raise it in the chat early; the Lead reassigns tasks rather than everything piling up at the end. |
| Groq rate limits | Everyone uses their own free key while developing. |
| Windows vs Mac differences (`npx.cmd`, paths, shell commands) | Test on both before the check-in; `run_command` picks the shell based on the OS. |
