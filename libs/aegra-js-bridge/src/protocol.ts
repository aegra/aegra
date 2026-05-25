// JSON-RPC 2.0 types and serialization helpers

export interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: string | number;
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: string | number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

export interface JsonRpcNotification {
  jsonrpc: "2.0";
  method: string;
  params: Record<string, unknown>;
}

/** Build a successful JSON-RPC response. */
export function createResponse(
  id: string | number,
  result: unknown,
): JsonRpcResponse {
  return { jsonrpc: "2.0", id, result };
}

/** Build a JSON-RPC error response. */
export function createError(
  id: string | number,
  code: number,
  message: string,
  data?: unknown,
): JsonRpcResponse {
  return { jsonrpc: "2.0", id, error: { code, message, ...(data !== undefined && { data }) } };
}

/** Build a JSON-RPC notification (no id). */
export function createNotification(
  method: string,
  params: Record<string, unknown>,
): JsonRpcNotification {
  return { jsonrpc: "2.0", method, params };
}

/** Parse and validate a JSON-RPC request from a raw line. */
export function parseRequest(line: string): JsonRpcRequest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch {
    throw new Error("Invalid JSON");
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("Request must be a JSON object");
  }

  const obj = parsed as Record<string, unknown>;

  if (obj.jsonrpc !== "2.0") {
    throw new Error('Missing or invalid "jsonrpc" field (must be "2.0")');
  }

  if (obj.id === undefined || obj.id === null) {
    throw new Error('Missing "id" field');
  }

  if (typeof obj.id !== "string" && typeof obj.id !== "number") {
    throw new Error('"id" must be a string or number');
  }

  if (typeof obj.method !== "string" || obj.method.length === 0) {
    throw new Error('"method" must be a non-empty string');
  }

  if (obj.params !== undefined) {
    if (typeof obj.params !== "object" || obj.params === null || Array.isArray(obj.params)) {
      throw new Error('"params" must be an object if provided');
    }
  }

  return {
    jsonrpc: "2.0",
    id: obj.id as string | number,
    method: obj.method as string,
    params: obj.params as Record<string, unknown> | undefined,
  };
}
