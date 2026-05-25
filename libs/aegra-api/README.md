# aegra-api

Aegra API - Self-hosted Agent Protocol server.

Aegra is an open-source, self-hosted alternative to LangSmith Deployments. This package provides the core API server that implements the Agent Protocol, allowing you to run AI agents on your own infrastructure without vendor lock-in.

## Features

- **Agent Protocol Compliant**: Works with Agent Chat UI, LangGraph Studio, CopilotKit
- **MCP Support**: Exposes agents as MCP tools for Claude Desktop, Cursor, and other MCP clients
- **Drop-in Replacement**: Compatible with the LangGraph SDK
- **Self-Hosted**: Run on your own PostgreSQL database
- **Streaming Support**: Real-time streaming of agent responses
- **Human-in-the-Loop**: Built-in support for human approval workflows
- **Vector Store**: Semantic search capabilities with PostgreSQL

## Installation

```bash
pip install aegra-api
```

## Quick Start

The easiest way to get started is with the [aegra-cli](../aegra-cli/README.md):

```bash
# Install the CLI
pip install aegra-cli

# Initialize a new project (interactive)
aegra init
cd <your-project>

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Install dependencies and start developing
uv sync
uv run aegra dev
```

### Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
pip install aegra-api

# Set environment variables
export POSTGRES_USER=aegra
export POSTGRES_PASSWORD=aegra_secret
export POSTGRES_HOST=localhost
export POSTGRES_DB=aegra

# Run migrations
alembic upgrade head

# Start server
uvicorn aegra_api.main:app --port 2026 --reload
```

## Configuration

### aegra.json

Define your agent graphs in `aegra.json`:

```json
{
  "graphs": {
    "agent": "./graphs/my_agent/graph.py:graph",
    "assistant": "./graphs/assistant/graph.py:graph"
  },
  "http": {
    "app": "./custom_routes.py:app"
  }
}
```

### Environment Variables

```bash
# Database
POSTGRES_USER=aegra
POSTGRES_PASSWORD=aegra_secret
POSTGRES_HOST=localhost
POSTGRES_DB=aegra

# Authentication
AUTH_TYPE=noop  # Options: noop, custom

# Server
HOST=0.0.0.0
PORT=2026

# Configuration
AEGRA_CONFIG=aegra.json

# LLM (for example agents)
OPENAI_API_KEY=sk-...

# Observability (optional)
OTEL_TARGETS=LANGFUSE,PHOENIX
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/assistants` | POST | Create assistant from graph_id |
| `/assistants/search` | POST | Search assistants |
| `/assistants/{assistant_id}` | GET | Get assistant details |
| `/threads` | POST | Create conversation thread |
| `/threads/{thread_id}/state` | GET | Get thread state |
| `/threads/{thread_id}/runs` | POST | Execute graph (background) |
| `/threads/{thread_id}/runs/stream` | POST | Execute graph (streaming) |
| `/threads/{thread_id}/runs/{run_id}/cancel` | POST | Cancel a run |
| `/store/items` | PUT | Save to vector store |
| `/store/items/search` | POST | Semantic search |
| `/store/namespaces` | POST | List store namespaces |
| `/health` | GET | Health check |
| `/mcp` | POST | MCP (Model Context Protocol) Streamable HTTP endpoint |

## Creating Graphs

Agents are Python modules exporting a compiled `graph` variable:

```python
# graphs/my_agent/graph.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: list[str]

def process_node(state: State) -> State:
    messages = state.get("messages", [])
    messages.append("Processed!")
    return {"messages": messages}

# Build the graph
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# Export as 'graph'
graph = builder.compile()
```

## Architecture

```
+---------------------------------------------------------+
|  FastAPI HTTP Layer (Agent Protocol API)                |
|  - /assistants, /threads, /runs, /store endpoints       |
+---------------------------------------------------------+
|  Middleware Stack                                        |
|  - Auth, CORS, Structured Logging, Correlation ID       |
+---------------------------------------------------------+
|  Service Layer (Business Logic)                          |
|  - LangGraphService, AssistantService, StreamingService |
+---------------------------------------------------------+
|  LangGraph Runtime                                       |
|  - Graph execution, state management, tool execution    |
+---------------------------------------------------------+
|  Database Layer (PostgreSQL)                             |
|  - AsyncPostgresSaver (checkpoints), AsyncPostgresStore |
+---------------------------------------------------------+
```

## Package Structure

```
libs/aegra-api/
├── src/aegra_api/
│   ├── api/              # Agent Protocol endpoints
│   │   ├── assistants.py # /assistants CRUD
│   │   ├── threads.py    # /threads and state management
│   │   ├── runs.py       # /runs execution and streaming
│   │   └── store.py      # /store vector storage
│   ├── services/         # Business logic layer
│   ├── core/             # Infrastructure (database, auth, orm)
│   ├── models/           # Pydantic request/response schemas
│   ├── middleware/       # ASGI middleware
│   ├── observability/    # OpenTelemetry tracing
│   ├── utils/            # Helper functions
│   ├── main.py           # FastAPI app entry point
│   ├── config.py         # HTTP/store config loading
│   └── settings.py       # Environment settings
├── tests/                # Test suite
├── alembic/              # Database migrations
└── pyproject.toml
```

## Related Packages

- **aegra-cli**: Command-line interface for project management

## Documentation

For full documentation, see the [docs/](../../docs/) directory.
