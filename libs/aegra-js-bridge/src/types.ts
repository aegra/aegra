/** Metadata about a loaded graph, including its schemas. */
export interface GraphInfo {
  graphId: string;
  inputSchema: Record<string, unknown>;
  outputSchema: Record<string, unknown>;
  configSchema?: Record<string, unknown>;
}

/** A single event emitted during graph streaming. */
export interface StreamEvent {
  mode: string; // "values" | "updates" | "messages" | "debug"
  data: unknown;
}

/** The result of a synchronous graph invocation. */
export interface InvokeResult {
  state: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}
