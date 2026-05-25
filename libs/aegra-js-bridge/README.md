# @aegra/js-bridge

Node.js bridge that allows the Aegra Python API to execute [LangGraph.js](https://github.com/langchain-ai/langgraphjs) graphs via a JSON-RPC 2.0 protocol over stdio.

## How It Works

The Aegra API spawns this package as a child process. Communication happens over **stdin** (requests) and **stdout** (responses). Diagnostic logs go to **stderr** so they never interfere with the protocol.

```text
┌──────────────┐  stdin (JSON-RPC)   ┌──────────────────┐
│  Aegra API   │ ──────────────────► │  aegra-js-bridge  │
│  (Python)    │ ◄────────────────── │  (Node.js / tsx)  │
└──────────────┘  stdout (JSON-RPC)  └──────────────────┘
```

On startup the bridge emits a `ready` notification. The host can then load graph files, invoke them, or stream results.

## Quick Start

```bash
# Install dependencies
npm install

# Run the bridge (used by Aegra API, not typically run directly)
npm start

# Build TypeScript to dist/
npm run build

# Run tests
npm test
```

## JSON-RPC Protocol

Every message is a single line of JSON terminated by `\n`.

### Requests (host → bridge)

| Method        | Params                                              | Description                    |
|---------------|-----------------------------------------------------|--------------------------------|
| `ping`        | —                                                   | Health check                   |
| `load_graph`  | `path`, `export_name`, `graph_id`                   | Load a graph file and cache it |
| `get_schema`  | `graph_id`                                          | Return cached schema info      |
| `invoke`      | `graph_id`, `input`, `config?`                      | Run graph synchronously        |
| `stream`      | `graph_id`, `input`, `config?`, `stream_mode?`      | Stream graph execution events  |
| `shutdown`    | —                                                   | Gracefully shut down           |

### Responses (bridge → host)

Successful responses include a `result` field; failures include an `error` field with `code`, `message`, and optional `data`.

During a `stream` call the bridge sends `stream_event` **notifications** (no `id`) for each event, followed by a final response with `{ "status": "complete" }`.

### Example

```jsonc
// Request
{"jsonrpc":"2.0","id":1,"method":"ping"}

// Response
{"jsonrpc":"2.0","id":1,"result":{"status":"ok"}}
```

## Adding a New Graph

1. Create a `.ts` or `.js` file that exports a `StateGraph` (or a pre-compiled graph):

   ```typescript
   import { StateGraph, Annotation } from "@langchain/langgraph";

   const MyState = Annotation.Root({ messages: Annotation<string[]>({ default: () => [] }) });

   const graph = new StateGraph(MyState)
     .addNode("greet", async (state) => ({ messages: [...state.messages, "Hello!"] }))
     .addEdge("__start__", "greet")
     .addEdge("greet", "__end__");

   export default graph;
   ```

2. Load it from the host by sending:

   ```json
   {"jsonrpc":"2.0","id":1,"method":"load_graph","params":{"path":"./my-graph.ts","export_name":"default","graph_id":"my-graph"}}
   ```

3. Then invoke or stream as needed.

## Project Structure

```text
src/
├── index.ts           # Entry point – stdin/stdout JSON-RPC server loop
├── protocol.ts        # JSON-RPC 2.0 types & serialization
├── graph-loader.ts    # Dynamic import of graph files, caching
├── graph-executor.ts  # invoke() and stream() wrappers
└── types.ts           # Shared TypeScript interfaces
```
