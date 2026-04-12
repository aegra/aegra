import * as readline from "node:readline";
import {
  type JsonRpcRequest,
  parseRequest,
  createResponse,
  createError,
  createNotification,
} from "./protocol.js";
import { loadGraph, getSchema } from "./graph-loader.js";
import { invoke, stream } from "./graph-executor.js";

// ---------------------------------------------------------------------------
// Transport helpers
// ---------------------------------------------------------------------------

/** Write a JSON-serialized object followed by a newline to stdout. */
function send(obj: unknown): void {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

/** Log diagnostic messages to stderr so they never pollute the protocol. */
function log(message: string): void {
  process.stderr.write(`[aegra-js-bridge] ${message}\n`);
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

async function handleRequest(request: JsonRpcRequest): Promise<void> {
  const { id, method, params } = request;

  try {
    switch (method) {
      case "ping": {
        send(createResponse(id, { status: "ok" }));
        break;
      }

      case "load_graph": {
        const {
          path: graphPath,
          export_name: exportName,
          graph_id: graphId,
        } = (params ?? {}) as Record<string, string>;

        if (!graphPath || !exportName || !graphId) {
          send(
            createError(
              id,
              -32602,
              'load_graph requires "path", "export_name", and "graph_id" params',
            ),
          );
          return;
        }

        const info = await loadGraph(graphPath, exportName, graphId);
        send(createResponse(id, info));
        break;
      }

      case "get_schema": {
        const { graph_id: graphId } = (params ?? {}) as Record<string, string>;
        if (!graphId) {
          send(createError(id, -32602, 'get_schema requires "graph_id" param'));
          return;
        }
        const schema = getSchema(graphId);
        send(createResponse(id, schema));
        break;
      }

      case "invoke": {
        const {
          graph_id: graphId,
          input,
          config,
        } = (params ?? {}) as {
          graph_id?: string;
          input?: Record<string, unknown>;
          config?: Record<string, unknown>;
        };

        if (!graphId || !input) {
          send(
            createError(
              id,
              -32602,
              'invoke requires "graph_id" and "input" params',
            ),
          );
          return;
        }

        const result = await invoke(graphId, input, config);
        send(createResponse(id, result));
        break;
      }

      case "stream": {
        const {
          graph_id: graphId,
          input,
          config,
          stream_mode: streamMode,
        } = (params ?? {}) as {
          graph_id?: string;
          input?: Record<string, unknown>;
          config?: Record<string, unknown>;
          stream_mode?: string[];
        };

        if (!graphId || !input) {
          send(
            createError(
              id,
              -32602,
              'stream requires "graph_id" and "input" params',
            ),
          );
          return;
        }

        for await (const event of stream(graphId, input, config, streamMode)) {
          send(
            createNotification("stream_event", {
              request_id: id,
              ...event,
            }),
          );
        }

        send(createResponse(id, { status: "complete" }));
        break;
      }

      case "shutdown": {
        send(createResponse(id, { status: "shutting_down" }));
        process.exit(0);
        break; // unreachable, keeps linter happy
      }

      default: {
        send(createError(id, -32601, `Method not found: ${method}`));
      }
    }
  } catch (error: unknown) {
    const err = error instanceof Error ? error : new Error(String(error));
    send(
      createError(id, -32000, err.message, {
        name: err.name,
        stack: err.stack,
      }),
    );
  }
}

// ---------------------------------------------------------------------------
// Stdin line reader
// ---------------------------------------------------------------------------

const rl = readline.createInterface({
  input: process.stdin,
  output: undefined, // don't echo back to stdout
  terminal: false,
});

rl.on("line", async (line: string) => {
  if (!line.trim()) return;

  try {
    const request = parseRequest(line);
    await handleRequest(request);
  } catch (error: unknown) {
    const err = error instanceof Error ? error : new Error(String(error));
    send(createError(0, -32700, `Parse error: ${err.message}`));
  }
});

rl.on("close", () => {
  log("stdin closed, exiting");
  process.exit(0);
});

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

process.on("SIGTERM", () => {
  log("received SIGTERM");
  process.exit(0);
});

process.on("SIGINT", () => {
  log("received SIGINT");
  process.exit(0);
});

// Prevent crashes from unhandled promise rejections
process.on("unhandledRejection", (reason: unknown) => {
  log(`unhandled rejection: ${reason}`);
});

// ---------------------------------------------------------------------------
// Ready signal
// ---------------------------------------------------------------------------

send(createNotification("ready", { version: "1.0.0" }));
log("bridge started, waiting for JSON-RPC requests on stdin");

// Keep the process alive while stdin is open
process.stdin.resume();
