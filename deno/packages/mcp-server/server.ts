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

  constructor(requestId: JsonRpcId = null) {
    this.requestId = requestId;
  }

  /** Mark this context as initialized. */
  setInitialized(): void {
    this.initialized = true;
  }
}

/**
 * MCP server that processes JSON-RPC requests via a transport.
 * Equivalent to Rust `McpServer`.
 */
export class McpServer {
  private handler: McpHandler;
  private middleware: MiddlewareChain;
  private transport: ServerTransport;

  constructor(handler: McpHandler) {
    this.handler = handler;
    this.middleware = new MiddlewareChain();
    this.transport = new StdioServerTransport();
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
    console.info("MCP Relay server starting");

    while (true) {
      let line: string | null;
      try {
        line = await this.transport.readRequest();
      } catch (e) {
        console.error(`Transport read error: ${e}`);
        break;
      }

      if (line === null) {
        console.debug("Transport closed (EOF)");
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

      // Run response middleware
      await this.middleware.processResponse(response, ctx);

      await this.writeResponse(response);
    }

    if (this.handler.onShutdown) {
      await this.handler.onShutdown();
    }
    console.info("MCP Relay server shut down");
  }

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

  private async handleInitialize(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcResponse> {
    const params = (request.params as InitializeParams) ?? {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "unknown", version: "0.5.0" },
    };

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

  private async writeResponse(response: JsonRpcResponse): Promise<void> {
    const json = JSON.stringify(response);
    await this.transport.writeResponse(json);
  }
}
