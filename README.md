# Corvus

**An autonomous coding assistant for your terminal.**

Give Corvus a task in plain English. It reads your code, makes changes, runs commands, looks things up in docs or on the web, checks the results, and keeps going until the task is done. Every tool call shows up on screen, and by default it asks before it writes a file or runs a command.

> **Project status: in development.** Planning docs were committed first (tag `planning-v1`). The agent loop, provider layer, and MCP client with the filesystem and Tavily servers work; the custom RAG server is in progress.

ITCS 5010 · Group Project 2: CLI Coding Assistant · Group 6

## Planning documents

- [Initial architecture (v1)](docs/ARCHITECTURE.md): components, diagram, design decisions, risks
- [Project plan](docs/PROJECT_PLAN.md): roles, interfaces, week-by-week timeline, check-in goals

![Corvus v1 architecture](docs/diagrams/architecture-v1.png)

## What it will do

- **Agentic loop:** reason → call tools → observe results → repeat, until the task is finished
- **Two LLM backends:** local models through Ollama, cloud models through Groq, switchable at runtime
- **Three MCP servers, loaded dynamically:**
  - Filesystem (`@modelcontextprotocol/server-filesystem`): read, write, edit, and search files
  - Tavily: web search
  - `docs-rag` (ours): searches the FastAPI documentation using vector search + cross-encoder **reranking**
- **Confirm or auto mode:** approve each file write and shell command, or let it run on its own
- **Readable terminal UI:** streaming responses, a panel for every tool call, status spinners

## Team

| Name                  | Role               |
| --------------------- | ------------------ |
| Soumyanil Ain         | Lead, agent core   |
| Sogol Maghzian        | MCP client + tools |
| Meghana Thummalapally | RAG server         |
| Sumiran Juthuga       | CLI + diagrams     |

## Planned tech stack

Python 3.11+, LangChain (`langchain-ollama`, `langchain-groq`), `mcp` Python SDK + `langchain-mcp-adapters`, ChromaDB, sentence-transformers, Rich, prompt_toolkit, Node.js 18+ (for the npx-based MCP servers).

## Setup

You need Python 3.11+ and Node.js 18+ (the filesystem MCP server runs through `npx`).

```powershell
git clone https://github.com/soumyanilain/corvus-cli.git
cd corvus-cli
python -m venv venv
venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # macOS/Linux: cp .env.example .env
```

Open `.env` and add your keys: `GROQ_API_KEY` (free at console.groq.com) and `TAVILY_API_KEY` (free at tavily.com).

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
| `/model groq` or `/model ollama [name]` | switch LLM provider at runtime |
| `/clear` | start a fresh conversation |
| `/exit` | quit |

Test only the MCP connections, without an LLM: `python -m mcp_tools.demo --workspace .`
