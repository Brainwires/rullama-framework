/**
 * @module server
 *
 * MCP server with JSON-RPC dispatch loop.
 * Equivalent to Rust's `McpServer`.
 */

import type {
  InitializeParams,
  InitializeResult,
  JsonRpcId,
  JsonRpcRequest,
  JsonRpcResponse,
} from "@rullama/mcp-client";
import type { McpHandler } from "./handler.ts";
import type { Middleware } from "./middleware/mod.ts";
import { MiddlewareChain } from "./middleware/mod.ts";
import type { ServerTransport } from "./transport/traits.ts";
import { StdioServerTransport } from "./transport/stdio.ts";
import { AgentNetworkError } from "./error.ts";

/**
 * Information about a connected MCP client.
 * Equivalent to Rust `ClientInfo`.
 */
export interface ClientInfo {
  /** Client name. */
  name: string;
  /** Client version. */
  version: string;
}

/**
 * Context for an MCP request.
 * Equivalent to Rust `RequestContext`.
 */
export class RequestContext {
  /** Connected client info, if available. */
  clientInfo: ClientInfo | null = null;
  /** JSON-RPC request ID. */
  requestId: JsonRpcId = null;
  /** Whether the connection has been initialized. */
  initialized = false;
  /** Arbitrary key-value metadata. */
  metadata: Map<string, unknown> = new Map();

  /**
   * Create an uninitialized context with no client info and empty metadata.
   * @param requestId Id of the JSON-RPC request being served (`null` before the first request).
   */
  constructor(requestId: JsonRpcId = null) {
    this.requestId = requestId;
  }

  /** Mark this context as initialized. */
  setInitialized(): void {
    this.initialized = true;
  }
}

/**
 * Structurally validate `initialize` params. A client that sends `{}` (or
 * nothing) previously crashed the serve loop on `params.clientInfo.name`.
 */
export function parseInitializeParams(raw: unknown): InitializeParams {
  const p = (typeof raw === "object" && raw !== null ? raw : {}) as Record<
    string,
    unknown
  >;
  const client =
    (typeof p.clientInfo === "object" && p.clientInfo !== null
      ? p.clientInfo
      : {}) as Record<string, unknown>;
  return {
    protocolVersion: typeof p.protocolVersion === "string"
      ? p.protocolVersion
      : "2024-11-05",
    capabilities:
      (typeof p.capabilities === "object" && p.capabilities !== null
        ? p.capabilities
        : {}) as InitializeParams["capabilities"],
    clientInfo: {
      name: typeof client.name === "string" ? client.name : "unknown",
      version: typeof client.version === "string" ? client.version : "unknown",
    },
  };
}

/**
 * MCP server that processes JSON-RPC requests via a transport: reads
 * newline-delimited requests, runs the middleware chain, dispatches
 * `initialize`, `notifications/initialized`, `tools/list` and `tools/call`
 * to the {@link McpHandler}, and writes responses back (never for notifications).
 * Equivalent to Rust `McpServer`.
 */
export class McpServer {
  private handler: McpHandler;
  private middleware: MiddlewareChain;
  private transport: ServerTransport;

  /**
   * Create a server with an empty middleware chain.
   * @param handler The application handler.
   * @param transport Transport to serve on (default: stdio). Passing one here
   *   avoids opening stdin when the server will never use it (tests, embedding).
   */
  constructor(handler: McpHandler, transport?: ServerTransport) {
    this.handler = handler;
    this.middleware = new MiddlewareChain();
    this.transport = transport ?? new StdioServerTransport();
  }

  /** Set a custom transport. Returns this for chaining. */
  withTransport(transport: ServerTransport): this {
    this.transport = transport;
    return this;
  }

  /** Add a middleware to the processing pipeline. Returns this for chaining. */
  withMiddleware(mw: Middleware): this {
    this.middleware.add(mw);
    return this;
  }

  /** Run the server event loop until the transport closes. */
  async run(): Promise<void> {
    const ctx = new RequestContext(null);
    console.error("MCP server starting");

    while (true) {
      let line: string | null;
      try {
        line = await this.transport.readRequest();
      } catch (e) {
        console.error(`Transport read error: ${e}`);
        break;
      }

      if (line === null) {
        console.error("Transport closed (EOF)");
        break;
      }

      let request: JsonRpcRequest;
      try {
        request = JSON.parse(line) as JsonRpcRequest;
      } catch (e) {
        const error = AgentNetworkError.parseError(String(e));
        const response: JsonRpcResponse = {
          jsonrpc: "2.0",
          id: null,
          error: error.toJsonRpcError(),
        };
        await this.writeResponse(response);
        continue;
      }

      ctx.requestId = request.id;

      // Run middleware chain
      const middlewareError = await this.middleware.processRequest(
        request,
        ctx,
      );
      if (middlewareError) {
        const response: JsonRpcResponse = {
          jsonrpc: "2.0",
          id: request.id,
          error: middlewareError,
        };
        await this.writeResponse(response);
        continue;
      }

      // Dispatch to handler
      const response = await this.handleRequest(request, ctx);

      // JSON-RPC: a notification (no id) MUST NOT be answered.
      if (request.id === undefined || request.id === null) continue;

      // Run response middleware
      await this.middleware.processResponse(response, ctx);

      await this.writeResponse(response);
    }

    if (this.handler.onShutdown) {
      await this.handler.onShutdown();
    }
    console.error("MCP server shut down");
  }

  /** Route a request by `method`; unknown methods get a `MethodNotFound` error response. */
  // deno-lint-ignore require-await
  private async handleRequest(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcResponse> {
    switch (request.method) {
      case "initialize":
        return this.handleInitialize(request, ctx);
      case "notifications/initialized":
        return {
          jsonrpc: "2.0",
          id: request.id,
          result: {},
        };
      case "tools/list":
        return this.handleListTools(request);
      case "tools/call":
        return this.handleCallTool(request, ctx);
      default: {
        const error = AgentNetworkError.methodNotFound(request.method);
        return {
          jsonrpc: "2.0",
          id: request.id,
          error: error.toJsonRpcError(),
        };
      }
    }
  }

  /**
   * Handle `initialize`: record the client info on `ctx`, mark it initialized,
   * invoke the handler's optional `onInitialize` (failures are logged, not
   * returned), and reply with protocol version `2024-11-05` plus the
   * handler's capabilities and server info.
   */
  private async handleInitialize(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcResponse> {
    const params = parseInitializeParams(request.params);

    ctx.clientInfo = {
      name: params.clientInfo.name,
      version: params.clientInfo.version,
    };
    ctx.setInitialized();

    if (this.handler.onInitialize) {
      try {
        await this.handler.onInitialize(params);
      } catch (e) {
        console.error(`Handler onInitialize failed: ${e}`);
      }
    }

    const info = this.handler.serverInfo();
    const capabilities = this.handler.capabilities();

    const result: InitializeResult = {
      protocolVersion: "2024-11-05",
      capabilities,
      serverInfo: info,
    };

    return {
      jsonrpc: "2.0",
      id: request.id,
      result: result as unknown as Record<string, unknown>,
    };
  }

  /** Handle `tools/list`: reply with the handler's tools as `{ name, description, inputSchema }`. */
  private handleListTools(request: JsonRpcRequest): JsonRpcResponse {
    const toolDefs = this.handler.listTools();

    const tools = toolDefs.map((t) => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    }));

    return {
      jsonrpc: "2.0",
      id: request.id,
      result: { tools },
    };
  }

  /**
   * Handle `tools/call`: reject with `InvalidParams` when `params` or
   * `params.name` is missing, otherwise call the handler with
   * `params.arguments` (default `{}`); a thrown error becomes an `Internal`
   * error response.
   */
  private async handleCallTool(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcResponse> {
    const params = request.params as Record<string, unknown> | undefined;
    if (!params) {
      const error = AgentNetworkError.invalidParams(
        "Missing params for tools/call",
      );
      return {
        jsonrpc: "2.0",
        id: request.id,
        error: error.toJsonRpcError(),
      };
    }

    const toolName = params.name as string | undefined;
    if (!toolName) {
      const error = AgentNetworkError.invalidParams(
        "Missing 'name' in tools/call",
      );
      return {
        jsonrpc: "2.0",
        id: request.id,
        error: error.toJsonRpcError(),
      };
    }

    const args = (params.arguments as Record<string, unknown>) ?? {};

    try {
      const result = await this.handler.callTool(toolName, args, ctx);
      return {
        jsonrpc: "2.0",
        id: request.id,
        result: result as unknown as Record<string, unknown>,
      };
    } catch (e) {
      const error = AgentNetworkError.internal(String(e));
      return {
        jsonrpc: "2.0",
        id: request.id,
        error: error.toJsonRpcError(),
      };
    }
  }

  /** Serialize `response` to JSON and hand it to the transport. */
  private async writeResponse(response: JsonRpcResponse): Promise<void> {
    const json = JSON.stringify(response);
    await this.transport.writeResponse(json);
  }
}
