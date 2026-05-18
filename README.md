<p align="center">
  <img src="docs/images/banner.png" alt="Aegra banner" />
</p>

<h1 align="center">Aegra</h1>

<p align="center">
  <strong>Self-hosted LangSmith Deployments alternative. Your infrastructure, your rules.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/aegra-api/"><img src="https://img.shields.io/pypi/v/aegra-api?label=aegra-api&color=blue" alt="PyPI API"></a>
  <a href="https://pypi.org/project/aegra-cli/"><img src="https://img.shields.io/pypi/v/aegra-cli?label=aegra-cli&color=blue" alt="PyPI CLI"></a>
  <a href="https://github.com/aegra/aegra/actions/workflows/ci.yml"><img src="https://github.com/aegra/aegra/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://app.codecov.io/gh/aegra/aegra"><img src="https://codecov.io/gh/aegra/aegra/graph/badge.svg" alt="Codecov"></a>
</p>

<p align="center">
  <a href="https://github.com/aegra/aegra/stargazers"><img src="https://img.shields.io/github/stars/aegra/aegra" alt="GitHub stars"></a>
  <a href="https://github.com/aegra/aegra/blob/main/LICENSE"><img src="https://img.shields.io/github/license/aegra/aegra" alt="License"></a>
  <a href="https://discord.com/invite/D5M3ZPS25e"><img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://patreon.com/aegra"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=github-sponsors&logoColor=white" alt="Sponsor"></a>
</p>

---

Aegra is a drop-in replacement for LangSmith Deployments. Use the same LangGraph SDK, same APIs, but run it on your own infrastructure with PostgreSQL persistence.

**Works with:** [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) | [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) | [AG-UI / CopilotKit](https://github.com/CopilotKit/CopilotKit)

## 🚀 Quick Start

### Using the CLI (Recommended)

**Prerequisites:** Python 3.12+, Docker (for PostgreSQL)

```bash
pip install aegra-cli

# Initialize a new project — prompts for location, template, and name
aegra init

# Follow the printed next steps:
cd <your-project>
cp .env.example .env     # Add your OPENAI_API_KEY to .env
uv sync                  # Install dependencies
uv run aegra dev         # Start PostgreSQL + dev server
```

> **Note:** Always install `aegra-cli` directly — not the `aegra` meta-package. The `aegra` package on PyPI is a convenience wrapper that does not support version pinning.

### From Source

```bash
git clone https://github.com/aegra/aegra.git
cd aegra
cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker compose up
```

Open [http://localhost:2026/docs](http://localhost:2026/docs) to explore the API.

Your existing LangGraph code works without changes:

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2026")

assistant = await client.assistants.create(graph_id="agent")
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={"messages": [{"type": "human", "content": "Hello!"}]},
):
    print(chunk)
```

## 🔥 Why Aegra?

*Based on [LangChain pricing](https://www.langchain.com/pricing) as of February 2026. An enterprise tier with self-hosting is also available at custom pricing.*

| | LangSmith Deployments | Aegra |
|:--|:--|:--|
| **Deploy agents** | Local dev only (Free), paid cloud (Plus) | Free, unlimited |
| **Custom auth** | Not available (Free), available (Plus) | Python handlers (JWT/OAuth/Firebase) |
| **Self-hosted** | Enterprise only (license key required) | Always (Apache 2.0) |
| **Own database** | Managed only (Free/Plus), bring your own (Enterprise) | Bring your own Postgres |
| **Tracing** | LangSmith only | Any OTLP backend (Langfuse, Phoenix, etc.) |
| **Data residency** | LangChain cloud (Free/Plus), your infrastructure (Enterprise) | Your infrastructure |
| **SDK** | LangGraph SDK | Same LangGraph SDK |

## ✨ Features

- **[Agent Protocol](https://github.com/langchain-ai/agent-protocol) compliant** - Works with Agent Chat UI, LangGraph Studio, CopilotKit
- **[MCP support](https://modelcontextprotocol.io/)** - Use deployed agents as tools in Claude Desktop, Cursor, and any MCP client
- **[Worker architecture](https://docs.aegra.dev/guides/worker-architecture)** - Redis job queue with 30 concurrent runs per instance, lease-based crash recovery, and horizontal scaling across multiple instances
- **[Human-in-the-loop](https://docs.aegra.dev/guides/human-in-the-loop)** - Approval gates and user intervention points
- **[Streaming](https://docs.aegra.dev/guides/streaming)** - Real-time SSE streaming with cross-instance pub/sub and automatic reconnection with event replay
- **[Persistent state](https://docs.aegra.dev/guides/threads-and-state)** - PostgreSQL checkpoints via LangGraph
- **[Configurable auth](https://docs.aegra.dev/guides/authentication)** - JWT, OAuth, Firebase, or none
- **[Unified Observability](https://docs.aegra.dev/guides/observability)** - Fan-out tracing support via OpenTelemetry
- **[Semantic store](https://docs.aegra.dev/guides/semantic-store)** - Vector embeddings with pgvector
- **[Custom routes](https://docs.aegra.dev/guides/custom-routes)** - Add your own FastAPI endpoints

## 🛠️ CLI Commands

```bash
aegra init              # Interactive — asks for location, template, and name
aegra init ./my-agent   # Create at path (still prompts for template)

aegra dev               # Start development server (hot reload + auto PostgreSQL)
aegra serve             # Start production server (no reload)
aegra up                # Build and start all Docker services
aegra down              # Stop Docker services

aegra version           # Show version info
```

## 📚 Documentation

**[docs.aegra.dev](https://docs.aegra.dev)** — Full documentation with guides, API reference, and configuration.

| Topic | Description |
|-------|-------------|
| [Quickstart](https://docs.aegra.dev/quickstart) | Get a running server in under 5 minutes |
| [Configuration](https://docs.aegra.dev/reference/configuration) | aegra.json format and all options |
| [Authentication](https://docs.aegra.dev/guides/authentication) | JWT, OAuth, Firebase, or custom auth handlers |
| [Worker Architecture](https://docs.aegra.dev/guides/worker-architecture) | Redis job queue, crash recovery, horizontal scaling |
| [Streaming](https://docs.aegra.dev/guides/streaming) | 8 SSE stream modes with reconnection |
| [Store](https://docs.aegra.dev/guides/store) | Key-value and semantic search storage |
| [Observability](https://docs.aegra.dev/guides/observability) | Fan-out tracing to Langfuse, Phoenix, or any OTLP backend |
| [Deployment](https://docs.aegra.dev/guides/deployment) | Docker, PaaS, and Kubernetes deployment |
| [Migration](https://docs.aegra.dev/migration) | Migrate from LangSmith Deployments |

## 💬 Community & Support

- **[Discord](https://discord.com/invite/D5M3ZPS25e)** - Chat with the community
- **[GitHub Discussions](https://github.com/aegra/aegra/discussions)** - Ask questions, share ideas
- **[GitHub Issues](https://github.com/aegra/aegra/issues)** - Report bugs

## 🏗️ Built With

- [FastAPI](https://fastapi.tiangolo.com/) - HTTP layer
- [LangGraph](https://github.com/langchain-ai/langgraph) - State management & graph execution
- [PostgreSQL](https://www.postgresql.org/) - Persistence & checkpoints
- [Redis](https://redis.io/) - Job queue, SSE pub/sub, crash recovery
- [OpenTelemetry](https://opentelemetry.io/) - Observability standard
- [pgvector](https://github.com/pgvector/pgvector) - Vector embeddings

## 🤝 Contributing

We welcome contributions! See [Contributing guide](https://docs.aegra.dev/guides/contributing) and check out [good first issues](https://github.com/aegra/aegra/labels/good%20first%20issue).

## 💖 Support the Project

The best contribution is code, PRs, and bug reports - that's what makes open source thrive.

For those who want to support Aegra financially, whether you're using it in production or just believe in what we're building, you can [become a sponsor](https://patreon.com/aegra). Sponsorships help keep development active and the project healthy.

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE).

---

<p align="center">
  <strong>⭐ Star us if Aegra helps you escape vendor lock-in ⭐</strong>
</p>

<a href="https://www.star-history.com/#aegra/aegra&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=aegra/aegra&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=aegra/aegra&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=aegra/aegra&type=Date" />
  </picture>
</a>
