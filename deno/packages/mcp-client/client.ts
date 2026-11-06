/**
 * @module client
 *
 * MCP Client — manages connections to MCP servers.
 * Equivalent to Rust's `rullama-mcp/src/client.rs`.
 */

import type { McpServerConfig } from "./config.ts";
import {
  type ClientTransport,
  StdioTransport,
  Transport,
} from "./transport.ts";
import type {
  CallToolParams,
  CallToolResult,
  GetPromptParams,
  GetPromptResult,
  InitializeParams,
  InitializeResult,
  JsonRpcId,
  JsonRpcNotification,
  JsonRpcResponse,
  ListPromptsResult,
  ListResourcesResult,
  ListToolsResult,
  McpPrompt,
  McpResource,
  McpTool,
  ReadResourceParams,
  ReadResourceResult,
  ServerCapabilities,
  ServerInfo,
} from "./types.ts";
import { createJsonRpcNotification, createJsonRpcRequest } from "./types.ts";

/** The `notifications/cancelled` notification for `requestId` (MCP cancellation). */
export function cancellationNotification(
  requestId: number,
  reason?: string,
): JsonRpcNotification {
  return createJsonRpcNotification("notifications/cancelled", {
    requestId,
    ...(reason !== undefined ? { reason } : {}),
  });
}

/** The MCP protocol revision this client offers in `initialize`. */
export const LATEST_PROTOCOL_VERSION = "2025-06-18";

/**
 * Protocol revisions this client can talk to. The server answers `initialize`
 * with the revision it wants to use; anything outside this list is rejected.
 */
export const SUPPORTED_PROTOCOL_VERSIONS: readonly string[] = [
  "2025-06-18",
  "2025-03-26",
  "2024-11-05",
];

/** Per-request options for {@link McpClient.request} and the typed helpers. */
export interface RequestOptions {
  /**
   * Abort the request: the promise rejects with the signal's reason and a
   * `notifications/cancelled` is sent so the server stops working on it.
   */
  signal?: AbortSignal;
  /** Receives every server notification that arrives while this request is pending. */
  onNotification?: (notification: JsonRpcNotification) => void;
}

/** Handler for server notifications on any connection (see {@link McpClient.onNotification}). */
export type NotificationHandler = (
  serverName: string,
  notification: JsonRpcNotification,
) => void;

interface Pending {
  resolve: (response: JsonRpcResponse) => void;
  reject: (error: unknown) => void;
}

/**
 * Active connection to an MCP server.
 * Equivalent to Rust `McpConnection`.
 */
interface McpConnection {
  /** Server name from config. */
  serverName: string;
  /** Transport layer. */
  transport: Transport;
  /** Server info from initialization. */
  serverInfo: ServerInfo;
  /** Server capabilities from initialization. */
  capabilities: ServerCapabilities;
  /** Protocol revision the server answered `initialize` with. */
  protocolVersion: string;
  /** In-flight requests by id. */
  pending: Map<string, Pending>;
  /** Per-request notification listeners (see {@link RequestOptions.onNotification}). */
  listeners: Set<(notification: JsonRpcNotification) => void>;
  /** Set once the connection is unusable; new requests reject with it. */
  closedError: Error | null;
  /** The background reader; settles when the connection ends. */
  reader: Promise<void>;
}

const idKey = (id: JsonRpcId): string => JSON.stringify(id);

/**
 * MCP Client — manages connections to MCP servers.
 * Equivalent to Rust `McpClient`.
 *
 * Provides methods to connect to MCP servers, list/call tools,
 * list/read resources, and list/get prompts.
 *
 * Each connection has a background reader that matches responses to requests
 * by `id`, so any number of calls may be in flight at once on one connection
 * and their responses may arrive in any order. Server notifications go to
 * {@link onNotification} handlers (and to a pending call's own
 * `onNotification`); a server `ping` is answered automatically.
 */
export class McpClient {
  #connections: Map<string, McpConnection> = new Map();
  #requestId = 1;
  #clientName: string;
  #clientVersion: string;
  #handlers = new Set<NotificationHandler>();

  /**
   * Create a new MCP client with the given name and version.
   * Equivalent to Rust `McpClient::new`.
   */
  constructor(clientName: string, clientVersion: string) {
    this.#clientName = clientName;
    this.#clientVersion = clientVersion;
  }

  /**
   * Create a default MCP client.
   * Equivalent to Rust `McpClient::default`.
   */
  static createDefault(): McpClient {
    return new McpClient("rullama", "0.5.0");
  }

  /**
   * Connect to an MCP server.
   * Equivalent to Rust `McpClient::connect`.
   */
  async connect(config: McpServerConfig): Promise<void> {
    // Spawn server process
    const stdioTransport = await StdioTransport.create(
      config.command,
      config.args,
      config.env,
    );
    await this.connectTransport(config.name, stdioTransport);
  }

  /**
   * Connect to an MCP server over an already-open transport (e.g.
   * `await SocketTransport.connectUnix(path)`) and run the `initialize`
   * handshake. On failure the transport is closed and the error rethrown.
   * @param serverName Handle every later call uses.
   * @param transport The connected transport.
   */
  async connectTransport(
    serverName: string,
    transport: ClientTransport,
  ): Promise<void> {
    const wrapped = transport instanceof Transport
      ? transport
      : new Transport(transport);
    const connection: McpConnection = {
      serverName,
      transport: wrapped,
      serverInfo: { name: "", version: "" },
      capabilities: {},
      protocolVersion: "",
      pending: new Map(),
      listeners: new Set(),
      closedError: null,
      reader: Promise.resolve(),
    };
    connection.reader = this.#readLoop(connection);

    try {
      const result = await this.#initialize(connection);
      connection.serverInfo = result.serverInfo;
      connection.capabilities = result.capabilities;
      connection.protocolVersion = result.protocolVersion;
    } catch (e) {
      this.#shutdown(connection, new Error("initialize failed"));
      await wrapped.close().catch(() => {});
      throw e;
    }

    this.#connections.set(serverName, connection);
  }

  /**
   * Disconnect from an MCP server.
   * Equivalent to Rust `McpClient::disconnect`.
   */
  async disconnect(serverName: string): Promise<void> {
    const connection = this.#connections.get(serverName);
    if (connection) {
      this.#connections.delete(serverName);
      this.#shutdown(connection, new Error(`Disconnected from ${serverName}`));
      await connection.transport.close();
      await connection.reader;
    }
  }

  /**
   * Check if connected to a server.
   * Equivalent to Rust `McpClient::is_connected`.
   */
  isConnected(serverName: string): boolean {
    return this.#connections.has(serverName);
  }

  /**
   * Get list of connected servers.
   * Equivalent to Rust `McpClient::list_connected`.
   */
  listConnected(): string[] {
    return [...this.#connections.keys()];
  }

  /**
   * Subscribe to server notifications from every connection (e.g.
   * `notifications/tools/list_changed`, `notifications/message`,
   * `notifications/progress`). Returns an unsubscribe function.
   */
  onNotification(handler: NotificationHandler): () => void {
    this.#handlers.add(handler);
    return () => this.#handlers.delete(handler);
  }

  /**
   * Send any JSON-RPC request and resolve with its `result`. A JSON-RPC
   * error rejects as `Error("<method> failed: <message> (code: <code>)")`.
   * @param serverName Connection to use.
   * @param method JSON-RPC method.
   * @param params Optional params.
   * @param options Cancellation signal and per-request notification listener.
   */
  async request(
    serverName: string,
    method: string,
    params?: unknown,
    options: RequestOptions = {},
  ): Promise<unknown> {
    const connection = this.#getConnection(serverName);
    const response = await this.#send(connection, method, params, options);
    if (response.error) {
      throw new Error(
        `${method} failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }
    return response.result;
  }

  /** Ping the server (`ping`); resolves when it answers. */
  async ping(serverName: string, options: RequestOptions = {}): Promise<void> {
    await this.request(serverName, "ping", undefined, options);
  }

  /**
   * List available tools from a server.
   * Equivalent to Rust `McpClient::list_tools`.
   */
  async listTools(
    serverName: string,
    options: RequestOptions = {},
  ): Promise<McpTool[]> {
    const result = await this.request(
      serverName,
      "tools/list",
      undefined,
      options,
    ) as ListToolsResult;
    return result.tools;
  }

  /**
   * Call a tool on a server.
   * Equivalent to Rust `McpClient::call_tool`.
   */
  async callTool(
    serverName: string,
    toolName: string,
    args?: Record<string, unknown>,
    options: RequestOptions = {},
  ): Promise<CallToolResult> {
    const params: CallToolParams = {
      name: toolName,
      ...(args ? { arguments: args } : {}),
    };
    return await this.request(
      serverName,
      "tools/call",
      params,
      options,
    ) as CallToolResult;
  }

  /**
   * Call a tool on a server with notification forwarding.
   * If onNotification is provided, any notifications received while waiting
   * for the response will be forwarded through that callback.
   * Equivalent to Rust `McpClient::call_tool_with_notifications`.
   */
  async callToolWithNotifications(
    serverName: string,
    toolName: string,
    args?: Record<string, unknown>,
    onNotification?: (notification: JsonRpcNotification) => void,
  ): Promise<CallToolResult> {
    return await this.callTool(serverName, toolName, args, { onNotification });
  }

  /**
   * List available resources from a server.
   * Equivalent to Rust `McpClient::list_resources`.
   */
  async listResources(serverName: string): Promise<McpResource[]> {
    const result = await this.request(
      serverName,
      "resources/list",
    ) as ListResourcesResult;
    return result.resources;
  }

  /**
   * Read a resource from a server.
   * Equivalent to Rust `McpClient::read_resource`.
   */
  async readResource(
    serverName: string,
    uri: string,
  ): Promise<ReadResourceResult> {
    const params: ReadResourceParams = { uri };
    return await this.request(
      serverName,
      "resources/read",
      params,
    ) as ReadResourceResult;
  }

  /**
   * List available prompts from a server.
   * Equivalent to Rust `McpClient::list_prompts`.
   */
  async listPrompts(serverName: string): Promise<McpPrompt[]> {
    const result = await this.request(
      serverName,
      "prompts/list",
    ) as ListPromptsResult;
    return result.prompts;
  }

  /**
   * Get a prompt from a server.
   * Equivalent to Rust `McpClient::get_prompt`.
   */
  async getPrompt(
    serverName: string,
    promptName: string,
    args?: Record<string, unknown>,
  ): Promise<GetPromptResult> {
    const params: GetPromptParams = {
      name: promptName,
      ...(args ? { arguments: args } : {}),
    };
    return await this.request(
      serverName,
      "prompts/get",
      params,
    ) as GetPromptResult;
  }

  /**
   * Get server info for a connection.
   * Equivalent to Rust `McpClient::get_server_info`.
   */
  getServerInfo(serverName: string): ServerInfo {
    const connection = this.#getConnection(serverName);
    return { ...connection.serverInfo };
  }

  /**
   * Get server capabilities for a connection.
   * Equivalent to Rust `McpClient::get_capabilities`.
   */
  getCapabilities(serverName: string): ServerCapabilities {
    const connection = this.#getConnection(serverName);
    return { ...connection.capabilities };
  }

  /** The protocol revision negotiated with a server. */
  getProtocolVersion(serverName: string): string {
    return this.#getConnection(serverName).protocolVersion;
  }

  /**
   * Tell the server to stop working on an in-flight request.
   *
   * MCP cancellation is the `notifications/cancelled` notification (no `id`,
   * never answered) carrying the `requestId` to cancel. The previous
   * implementation sent `$/cancelRequest`, which is an LSP method that no MCP
   * server recognises. Prefer {@link RequestOptions.signal}, which also
   * settles the local promise.
   */
  async cancelRequest(
    serverName: string,
    requestId: number,
    reason?: string,
  ): Promise<void> {
    const connection = this.#getConnection(serverName);
    await connection.transport.sendRequest(
      cancellationNotification(requestId, reason),
    );
  }

  /**
   * Get the next request ID.
   * Equivalent to Rust `McpClient::next_request_id`.
   */
  #nextRequestId(): number {
    return this.#requestId++;
  }

  /**
   * Get a connection by server name, throwing if not connected.
   */
  #getConnection(serverName: string): McpConnection {
    const connection = this.#connections.get(serverName);
    if (!connection) {
      throw new Error(`Not connected to server: ${serverName}`);
    }
    return connection;
  }

  /** Send a request on `connection` and resolve with the response matching its id. */
  #send(
    connection: McpConnection,
    method: string,
    params: unknown,
    options: RequestOptions,
  ): Promise<JsonRpcResponse> {
    if (connection.closedError) return Promise.reject(connection.closedError);
    const { signal, onNotification } = options;
    if (signal?.aborted) return Promise.reject(signal.reason);

    const id = this.#nextRequestId();
    const key = idKey(id);
    return new Promise<JsonRpcResponse>((resolve, reject) => {
      const onAbort = () => {
        cleanup();
        reject(signal!.reason);
        connection.transport
          .sendRequest(
            cancellationNotification(
              id,
              signal!.reason instanceof Error
                ? signal!.reason.message
                : String(signal!.reason ?? "aborted"),
            ),
          )
          .catch(() => {});
      };
      const cleanup = () => {
        connection.pending.delete(key);
        if (onNotification) connection.listeners.delete(onNotification);
        signal?.removeEventListener("abort", onAbort);
      };
      connection.pending.set(key, {
        resolve: (r) => {
          cleanup();
          resolve(r);
        },
        reject: (e) => {
          cleanup();
          reject(e);
        },
      });
      if (onNotification) connection.listeners.add(onNotification);
      signal?.addEventListener("abort", onAbort, { once: true });
      connection.transport
        .sendRequest(createJsonRpcRequest(id, method, params))
        .catch((e) => {
          cleanup();
          reject(e);
        });
    });
  }

  /** Read messages until the connection ends, routing each by kind and id. */
  async #readLoop(connection: McpConnection): Promise<void> {
    // Start on the next microtask, after `connection.reader` is assigned.
    await Promise.resolve();
    while (connection.closedError === null) {
      let message;
      try {
        message = await connection.transport.receiveMessage();
      } catch (e) {
        if (e instanceof SyntaxError) continue; // a garbled line, not EOF
        this.#shutdown(
          connection,
          new Error(
            `MCP server ${connection.serverName} connection closed: ${
              e instanceof Error ? e.message : String(e)
            }`,
          ),
        );
        break;
      }

      if (message.type === "notification") {
        this.#dispatchNotification(connection, message.notification);
        continue;
      }
      const response = message.response as JsonRpcResponse & {
        method?: unknown;
      };
      if (typeof response.method === "string") {
        this.#answerServerRequest(connection, response.id, response.method);
        continue;
      }
      // Responses for unknown ids (e.g. cancelled requests) are dropped.
      connection.pending.get(idKey(response.id))?.resolve(response);
    }
  }

  /** Mark the connection dead and fail every in-flight request. */
  #shutdown(connection: McpConnection, error: Error): void {
    if (connection.closedError === null) connection.closedError = error;
    for (const pending of [...connection.pending.values()]) {
      pending.reject(connection.closedError);
    }
    connection.pending.clear();
  }

  #dispatchNotification(
    connection: McpConnection,
    notification: JsonRpcNotification,
  ): void {
    if (typeof notification.method !== "string") return; // e.g. an id:null error
    for (const listener of [...connection.listeners]) {
      try {
        listener(notification);
      } catch (e) {
        console.error(`MCP notification listener failed: ${e}`);
      }
    }
    for (const handler of [...this.#handlers]) {
      try {
        handler(connection.serverName, notification);
      } catch (e) {
        console.error(`MCP notification handler failed: ${e}`);
      }
    }
  }

  /** Answer a server→client request: `ping` succeeds, anything else is method-not-found. */
  #answerServerRequest(
    connection: McpConnection,
    id: JsonRpcId,
    method: string,
  ): void {
    const reply: JsonRpcResponse = method === "ping"
      ? { jsonrpc: "2.0", id, result: {} }
      : {
        jsonrpc: "2.0",
        id,
        error: { code: -32601, message: `Method not found: ${method}` },
      };
    connection.transport.sendRequest(reply).catch(() => {});
  }

  /**
   * Initialize handshake with server.
   * Equivalent to Rust `McpClient::initialize`.
   */
  async #initialize(connection: McpConnection): Promise<InitializeResult> {
    const params: InitializeParams = {
      protocolVersion: LATEST_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: {
        name: this.#clientName,
        version: this.#clientVersion,
      },
    };

    const response = await this.#send(connection, "initialize", params, {});

    if (response.error) {
      throw new Error(
        `Initialize failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }

    if (!response.result) {
      throw new Error("Missing result in initialize response");
    }

    const result = response.result as InitializeResult;
    if (!SUPPORTED_PROTOCOL_VERSIONS.includes(result.protocolVersion)) {
      throw new Error(
        `Server protocol version ${result.protocolVersion} is not supported (client supports ${
          SUPPORTED_PROTOCOL_VERSIONS.join(", ")
        })`,
      );
    }

    // Send initialized notification
    await connection.transport.sendRequest(
      createJsonRpcNotification("notifications/initialized"),
    );

    return result;
  }

  /** The client name. */
  get clientName(): string {
    return this.#clientName;
  }

  /** The client version. */
  get clientVersion(): string {
    return this.#clientVersion;
  }

  /** The current request ID counter value (for testing). */
  get currentRequestId(): number {
    return this.#requestId;
  }
}
