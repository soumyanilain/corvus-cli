# Corvus

**An autonomous coding assistant for your terminal.**

Give Corvus a task in plain English. It reads your code, makes changes, runs commands, looks things up in the FastAPI docs or on the web, checks the results, and keeps going until the task is done. Every tool call shows up on screen, and by default it asks before it writes a file or runs a command.

The name comes from *Corvus*, the crow genus. Crows are one of the few animals that make and use tools.

ITCS 5010 · Group Project 2: CLI Coding Assistant · Group 6

## What it does

- **Agentic loop:** reason → call tools → observe the results → repeat, until the task is done ([`corvus/agent.py`](corvus/agent.py))
- **Model-agnostic:** Groq (cloud) and Ollama (local) behind one provider layer, switchable at runtime with `/model` ([`corvus/providers.py`](corvus/providers.py))
- **Three MCP servers, loaded dynamically** from [`servers.json`](servers.json) at startup ([`mcp_tools/`](mcp_tools/)):

  | Server | What the agent gets |
  |---|---|
  | `filesystem` (official `@modelcontextprotocol/server-filesystem`) | read, write, edit, list and search files, sandboxed to the workspace |
  | `tavily` (external resource server) | web search and page extraction |
  | `docs-rag` (ours, [`rag_server/`](rag_server/)) | search over the FastAPI documentation with vector search + cross-encoder **reranking** |

- **Local tools:** `run_command` (shell) and `grep_code` (search inside files)
- **Confirm or auto mode:** approve each file write and shell command, or let it run on its own
- **Terminal UI:** streaming answers, a panel for every tool call, spinners while things run ([`corvus/cli.py`](corvus/cli.py))

## Setup

You need **Python 3.11+**, **Node.js 18+** (the filesystem server runs through `npx`) and **git**.

```powershell
git clone https://github.com/soumyanilain/corvus-cli.git
cd corvus-cli
python -m venv venv
venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # macOS/Linux: cp .env.example .env
```

Open `.env` and add your keys: `GROQ_API_KEY` (free at console.groq.com) and `TAVILY_API_KEY` (free at tavily.com).

### Build the docs index (once)

```powershell
python -m rag_server.ingest
```

This downloads the FastAPI docs, splits them into ~1,500 chunks, embeds them and stores them in `data/chroma`. It takes a few minutes the first time (it also downloads two small models). Every later Corvus session reuses the stored index. Run it again with `--force` to rebuild.

## Run

```powershell
python -m corvus                          # works on the current folder, asks before risky actions
python -m corvus --workspace D:\my-app    # work on another project folder
python -m corvus --auto                   # auto-execute mode: no confirmation prompts
python -m corvus --provider ollama        # use a local model (run `ollama pull llama3.1:8b` first)
```

Inside Corvus, type a task in plain English, or one of these commands:

| Command | What it does |
|---|---|
| `/tools` | list the tools loaded from each MCP server |
| `/mode confirm` or `/mode auto` | ask before writing files / running commands, or don't |
| `/model groq [name]` or `/model ollama [name]` | switch the LLM at runtime |
| `/clear` | start a fresh conversation |
| `/exit` | quit |

Settings such as the default model live in `.env` (see `.env.example`).

### Try it on the sample project

```powershell
Copy-Item -Recurse eval\sample_project D:\corvus-demo
python -m corvus --workspace D:\corvus-demo
```

Then ask, for example: *"Add a POST /books endpoint with Pydantic validation, make GET /books/{id} return 404 for missing books, check the FastAPI docs for how, add tests, and run them."*

## Evaluation

```powershell
python -m eval.rag_eval        # vector search vs. vector search + reranking, 25 questions
python -m eval.llm_compare     # two LLMs on the same coding task, graded by hidden tests
```

Results are written to [`eval/results/`](eval/results/).

## Project structure

```
corvus/        agent loop, provider layer, terminal UI          (python -m corvus)
mcp_tools/     MCP client, tool registry, confirmation gate, run_command, grep_code
rag_server/    custom RAG MCP server: ingest, retriever with reranking, server
eval/          RAG evaluation, LLM comparison, sample project, grading tests
docs/          planning docs, architecture (v1 and v2), diagrams, report
servers.json   which MCP servers to start
```

## Documentation

- [Project plan](docs/PROJECT_PLAN.md) and [initial architecture (v1)](docs/ARCHITECTURE.md), committed before implementation (tag `planning-v1`)
- [Final architecture (v2)](docs/ARCHITECTURE_v2.md): what changed and why, state and sequence diagrams
- [Report (PDF)](docs/report/Corvus_Report.pdf)

## Team

| Name                  | Role               |
| --------------------- | ------------------ |
| Soumyanil Ain         | Lead, agent core   |
| Sogol Maghzian        | MCP client + tools |
| Meghana Thummalapally | RAG server         |
| Sumiran Juthuga       | CLI + diagrams     |
