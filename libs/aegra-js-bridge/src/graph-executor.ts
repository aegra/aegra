import { getGraph } from "./graph-loader.js";
import type { InvokeResult, StreamEvent } from "./types.js";

/**
 * Synchronously invoke a loaded graph and return the final state.
 */
export async function invoke(
  graphId: string,
  input: Record<string, unknown>,
  config?: Record<string, unknown>,
): Promise<InvokeResult> {
  const graph = await getGraph(graphId);
  const result = await graph.invoke(input, config);
  return { state: result };
}

/**
 * Stream events from a loaded graph.
 *
 * Yields `StreamEvent` objects as the graph progresses. The caller is
 * responsible for serializing each event and sending it to the client.
 */
export async function* stream(
  graphId: string,
  input: Record<string, unknown>,
  config?: Record<string, unknown>,
  streamMode?: string[],
): AsyncGenerator<StreamEvent> {
  const graph = await getGraph(graphId);

  const modes = streamMode ?? ["values"];

  const streamIterator = await graph.stream(input, {
    ...config,
    streamMode: modes,
  });

  // When multiple stream modes are requested LangGraph may yield tuples
  // of [mode, data]. When a single mode is used it yields the data directly.
  const singleMode = modes.length === 1;

  for await (const event of streamIterator) {
    if (singleMode) {
      yield { mode: modes[0], data: event };
    } else if (Array.isArray(event) && event.length === 2) {
      const [mode, data] = event as [string, unknown];
      yield { mode, data };
    } else {
      yield { mode: "values", data: event };
    }
  }
}
