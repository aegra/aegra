# MCP (Model Context Protocol)

Aegra exposes all configured graphs as MCP tools via the [Model Context Protocol](https://modelcontextprotocol.io/). This lets you use your deployed agents as tools in Claude Desktop, Cursor, and any other MCP-compatible client.

## Overview

When Aegra starts, it automatically creates one MCP tool per graph defined in `aegra.json`. The tool name matches the `graph_id`, and the input schema is auto-discovered from the graph's input schema.

Each tool invocation runs the agent and returns its output. The transport is Streamable HTTP, which supports both request/response and streaming interactions.

## Endpoint

```text
POST /mcp
```

Aegra exposes a single Streamable HTTP endpoint at `/mcp`. All MCP interactions go through this endpoint using the standard MCP JSON-RPC protocol over HTTP.

## Connecting from Claude Desktop

Add the following to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "aegra": {
      "url": "http://localhost:2026/mcp",
      "transport": "streamable-http"
    }
  }
}
```

After restarting Claude Desktop, your Aegra agents will appear as available tools.

## Connecting from Python

Using the FastMCP client:

```python
from fastmcp import Client

async with Client("http://localhost:2026/mcp") as client:
    tools = await client.list_tools()
    print(tools)
```

Or using the lower-level MCP SDK:

```python
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

async with streamable_http_client(url="http://localhost:2026/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print(tools)
```

## Tool discovery

Each graph becomes one MCP tool:

- **Tool name**: the `graph_id` from `aegra.json` (e.g., `"agent"`, `"assistant"`)
- **Input schema**: auto-derived from the graph's input schema
- **Description**: the graph's `description` field if set in config, otherwise `"Run the {graph_id} agent"`

For example, if `aegra.json` defines:

```json
{
  "graphs": {
    "agent": "./src/agent/graph.py:graph",
    "researcher": "./src/researcher/graph.py:graph"
  }
}
```

Then two MCP tools are exposed: `agent` and `researcher`.

## Authentication

MCP uses the same authentication as the rest of Aegra. If you have an `auth` handler configured in `aegra.json`, all MCP requests go through it.

For Claude Desktop and other clients that support HTTP headers, pass your token in the `Authorization` header:

```json
{
  "mcpServers": {
    "aegra": {
      "url": "http://localhost:2026/mcp",
      "transport": "streamable-http",
      "headers": {
        "Authorization": "Bearer <your-token>"
      }
    }
  }
}
```

If no auth is configured, all MCP requests are allowed.

### MCP auth provider (spec-compliant OAuth)

For MCP clients that support the [MCP OAuth flow](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization) (Dynamic Client Registration, authorization redirects, token exchange), Aegra can delegate authentication to any [FastMCP auth provider](https://gofastmcp.com/servers/auth).

Create a Python file that exports a FastMCP auth provider, then point to it via `mcp.auth.path` in `aegra.json`:

**aegra.json:**

```json
{
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "mcp": {
    "auth": {
      "path": "./mcp_auth.py:mcp_auth"
    }
  }
}
```

**mcp_auth.py** (example using OIDCProxy):

```python
import os
from fastmcp.server.auth.oidc_proxy import OIDCProxy

mcp_auth = OIDCProxy(
    config_url="https://your-idp.com/.well-known/openid-configuration",
    client_id="your-client-id",
    client_secret=os.environ["MCP_OIDC_CLIENT_SECRET"],
    base_url="http://localhost:2026/mcp",
)
```

Because the auth provider is a Python file, you can use any FastMCP auth provider -- `OIDCProxy`, a custom implementation, or anything that implements the FastMCP auth provider interface. You also have full control over constructor parameters, including objects that cannot be expressed in JSON config (custom token verifiers, storage backends, etc.).

See [`examples/mcp_auth_example.py`](../examples/mcp_auth_example.py) for more configurations (PKCE clients, custom token verifiers, extra OAuth params).

When an auth provider is configured:

1. **FastMCP handles the OAuth flow** -- MCP clients authenticate through the standard OAuth dance.
2. **Aegra's `@auth.authenticate` handler has the final say** -- the validated upstream token is forwarded to your auth handler as `Authorization: Bearer <upstream-token>`. Your handler decides identity, permissions, and whether to allow the request.

This means the auth provider enables the token exchange flow, but never bypasses Aegra's auth. If no `mcp.auth.path` is configured, requests pass through as anonymous.

The `path` format supports the same conventions as `auth.path`:
- `./mcp_auth.py:mcp_auth` -- load from a file relative to `aegra.json`
- `./src/auth/mcp.py:provider` -- nested path
- `mypackage.mcp_auth:provider` -- load from an installed package

## Stateless operation

The MCP endpoint is stateless. Each tool call is an independent request — there are no persistent sessions. State between calls is not maintained at the MCP layer. If you need persistent conversation state, use the Agent Protocol (`/threads` and `/runs`) directly.

## Configuration

MCP is enabled by default. To disable it, set `disable_mcp` in the `http` section of `aegra.json`:

```json
{
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "http": {
    "disable_mcp": true
  }
}
```

See the [configuration reference](/reference/configuration) for all `http` options.

## Response filtering

By default, MCP tool calls return the **full graph state** — including all intermediate messages (tool calls, tool responses, reasoning steps). For agents with a `messages` list in their state, this can be verbose.

To return only the last AI message, enable `final_response_only` in the `mcp` section of `aegra.json`:

```json
{
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "mcp": {
    "final_response_only": true
  }
}
```

When enabled, the MCP adapter extracts the last AI message from the output and returns it as a single JSON object. If no AI message is found (e.g., non-message-based graphs), the full output is returned as a fallback.

| Setting | Default | Behavior |
|---------|---------|----------|
| `final_response_only: false` | Yes | Return full graph state (all messages, metadata) |
| `final_response_only: true` | No | Return only the last AI message |
