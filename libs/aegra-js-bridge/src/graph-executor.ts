import { Command } from "@langchain/langgraph";
import { getGraph } from "./graph-loader.js";
import type { InvokeResult, StreamEvent } from "./types.js";

/**
 * Reconstruct a LangGraph Command from a serialized wire format.
 *
 * The Python side serialises resume commands as:
 * `{ "__command__": { "resume": ..., "goto": ..., "update": ... } }`
 */
function maybeCommand(
  input: Record<string, unknown>,
): Command | Record<string, unknown> {
  const raw = input.__command__ as Record<string, unknown> | undefined;
  if (!raw) return input;

  return new Command({
    resume: raw.resume,
    goto: raw.goto as string | string[] | undefined,
    update: raw.update as Record<string, unknown> | undefined,
  });
}

/**
 * Synchronously invoke a loaded graph and return the final state.
 */
export async function invoke(
  graphId: string,
  input: Record<string, unknown>,
  config?: Record<string, unknown>,
): Promise<InvokeResult> {
  const graph = await getGraph(graphId);
  const resolved = maybeCommand(input);
  const result = await graph.invoke(resolved, config);
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
  const resolved = maybeCommand(input);

  const modes = streamMode ?? ["values"];

  const streamIterator = await graph.stream(resolved, {
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
