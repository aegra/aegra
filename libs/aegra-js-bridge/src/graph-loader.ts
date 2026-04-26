import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import type { GraphInfo } from "./types.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyCompiledGraph = any;

/** Cache of compiled graph instances keyed by graph_id. */
const graphCache = new Map<string, AnyCompiledGraph>();

/** Cache of schema info keyed by graph_id. */
const schemaCache = new Map<string, GraphInfo>();

/** Singleton checkpointer — created once, shared across all graphs. */
let checkpointerInstance: PostgresSaver | null = null;
let checkpointerSetupPromise: Promise<void> | null = null;

/**
 * Get or create the singleton PostgresSaver checkpointer.
 *
 * Uses DATABASE_URL from the environment (normalised to postgresql:// by
 * the parent Python process). The checkpointer's setup() is idempotent
 * (CREATE TABLE IF NOT EXISTS) so a mixed Python+JS project sharing one
 * Postgres database is safe.
 */
async function getCheckpointer(): Promise<PostgresSaver> {
  if (checkpointerInstance) {
    await checkpointerSetupPromise;
    return checkpointerInstance;
  }

  const dbUrl = process.env.DATABASE_URL;
  if (!dbUrl) {
    throw new Error(
      "DATABASE_URL environment variable is required for JS graph checkpointing. " +
        "Set it to a postgresql:// connection string.",
    );
  }

  checkpointerInstance = PostgresSaver.fromConnString(dbUrl);
  checkpointerSetupPromise = checkpointerInstance.setup();
  await checkpointerSetupPromise;
  return checkpointerInstance;
}

/**
 * Dynamically import a graph definition file, compile with the native
 * PostgresSaver checkpointer, cache it, and return schema information.
 */
export async function loadGraph(
  filePath: string,
  exportName: string,
  graphId: string,
): Promise<GraphInfo> {
  const resolvedPath = path.resolve(filePath);
  const fileUrl = pathToFileURL(resolvedPath).href;

  // Dynamic import – works for both .ts (via tsx) and .js files.
  // Append a cache-busting query so re-loads pick up changes.
  const mod = await import(`${fileUrl}?t=${Date.now()}`);

  const exported = mod[exportName];
  if (!exported) {
    throw new Error(
      `Export "${exportName}" not found in ${resolvedPath}. ` +
        `Available exports: ${Object.keys(mod).join(", ")}`,
    );
  }

  const checkpointer = await getCheckpointer();

  let compiled: AnyCompiledGraph;
  if (typeof exported.compile === "function") {
    // Uncompiled StateGraph — compile with native checkpointer
    compiled = exported.compile({ checkpointer });
  } else if (typeof exported.invoke === "function") {
    // Already compiled — cannot inject checkpointer after compilation.
    // Warn and use as-is; checkpointing features (interrupt, resume,
    // time-travel) will not work unless the user compiled with their own
    // checkpointer.
    process.stderr.write(
      `[aegra-js-bridge] WARNING: Graph "${graphId}" is already compiled. ` +
        "Checkpointing cannot be injected. Export an uncompiled StateGraph " +
        "for full interrupt/resume/time-travel support.\n",
    );
    compiled = exported;
  } else {
    throw new Error(
      `Export "${exportName}" is neither a StateGraph nor a compiled graph`,
    );
  }

  graphCache.set(graphId, compiled);

  // Extract schema information when available.
  const inputSchema = extractSchema(compiled, "input");
  const outputSchema = extractSchema(compiled, "output");
  const configSchema = extractSchema(compiled, "config");

  const info: GraphInfo = {
    graphId,
    inputSchema,
    outputSchema,
    ...(configSchema && { configSchema }),
  };

  schemaCache.set(graphId, info);
  return info;
}

/** Return a previously loaded compiled graph, or throw. */
export async function getGraph(graphId: string): Promise<AnyCompiledGraph> {
  const graph = graphCache.get(graphId);
  if (!graph) {
    throw new Error(
      `Graph "${graphId}" is not loaded. Call load_graph first.`,
    );
  }
  return graph;
}

/** Return cached schema info for a loaded graph. */
export function getSchema(graphId: string): GraphInfo {
  const info = schemaCache.get(graphId);
  if (!info) {
    throw new Error(
      `Schema for graph "${graphId}" not found. Call load_graph first.`,
    );
  }
  return info;
}

/** Clean up the checkpointer connection on shutdown. */
export async function shutdownCheckpointer(): Promise<void> {
  if (checkpointerInstance) {
    try {
      await checkpointerInstance.end();
    } catch {
      // Best-effort cleanup
    }
    checkpointerInstance = null;
    checkpointerSetupPromise = null;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractSchema(
  graph: AnyCompiledGraph,
  kind: "input" | "output" | "config",
): Record<string, unknown> {
  try {
    if (kind === "input" && typeof graph.getInputJsonSchema === "function") {
      return graph.getInputJsonSchema() as Record<string, unknown>;
    }
    if (kind === "output" && typeof graph.getOutputJsonSchema === "function") {
      return graph.getOutputJsonSchema() as Record<string, unknown>;
    }
    if (kind === "config" && typeof graph.getConfigJsonSchema === "function") {
      return graph.getConfigJsonSchema() as Record<string, unknown>;
    }
  } catch {
    // Schema extraction is best-effort.
  }
  return {};
}
