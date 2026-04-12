import * as path from "node:path";
import { pathToFileURL } from "node:url";
import type { GraphInfo } from "./types.js";

// We keep a loose type alias for compiled graphs so we don't hard-require
// @langchain/langgraph at import time (it's loaded dynamically).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyCompiledGraph = any;

/** Cache of compiled graph instances keyed by graph_id. */
const graphCache = new Map<string, AnyCompiledGraph>();

/** Cache of schema info keyed by graph_id. */
const schemaCache = new Map<string, GraphInfo>();

/**
 * Dynamically import a graph definition file, compile (if needed), cache it,
 * and return its schema information.
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

  // If the export has a .compile() method it's an uncompiled StateGraph.
  let compiled: AnyCompiledGraph;
  if (typeof exported.compile === "function") {
    compiled = exported.compile();
  } else if (typeof exported.invoke === "function") {
    // Already compiled
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractSchema(
  graph: AnyCompiledGraph,
  kind: "input" | "output" | "config",
): Record<string, unknown> {
  try {
    // LangGraph compiled graphs expose getInputJsonSchema / getOutputJsonSchema
    // or similar helpers depending on version.
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
