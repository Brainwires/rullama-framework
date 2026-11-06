/**
 * @module server
 *
 * MCP server with a concurrent JSON-RPC dispatch loop per connection.
 * Equivalent to Rust's `McpServer`.
 */

import {
  createJsonRpcNotification,
  type InitializeParams,
  type InitializeResult,
  type JsonRpcId,
  type JsonRpcRequest,
  type JsonRpcResponse,
  LATEST_PROTOCOL_VERSION,
  SUPPORTED_PROTOCOL_VERSIONS,
} from "@rullama/mcp-client";
import type { McpHandler } from "./handler.ts";
import type { Middleware } from "./middleware/mod.ts";
import { MiddlewareChain } from "./middleware/mod.ts";
import type { ConnectionInfo, ServerTransport } from "./transport/traits.ts";
import { StdioServerTransport } from "./transport/stdio.ts";
import {
  type AcceptedConnection,
  nextConnectionId,
} from "./transport/socket.ts";
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

/** Severity of a `notifications/message` log entry (RFC 5424 levels, as MCP uses them). */
export type LoggingLevel =
  | "debug"
  | "info"
  | "notice"
  | "warning"
  | "error"
  | "critical"
  | "alert"
  | "emergency";

const LOG_LEVELS: readonly LoggingLevel[] = [
  "debug",
  "info",
  "notice",
  "warning",
  "error",
  "critical",
  "alert",
  "emergency",
];

/** A signal that never aborts, for contexts created outside a session. */
const NEVER = new AbortController().signal;

/**
 * Context for an MCP request.
 * Equivalent to Rust `RequestContext`.
 *
 * Inside a server session every request gets its own context: `requestId`
 * and `signal` are per request, while `clientInfo`, `initialized`,
 * `protocolVersion`, `connection` and the `metadata` map are the
 * connection's (the map is shared by every request on that connection).
 */
export class RequestContext {
  /** Connected client info, if available. */
  clientInfo: ClientInfo | null = null;
  /** JSON-RPC request ID. */
  requestId: JsonRpcId = null;
  /** Whether the connection has been initialized. */
  initialized = false;
  /** Arbitrary key-value metadata (connection-scoped inside a session). */
  metadata: Map<string, unknown> = new Map();
  /**
   * Aborted when the client cancels this request (`notifications/cancelled`)
   * or the connection goes away. Long-running tools should honour it.
   */
  signal: AbortSignal = NEVER;
  /** Which connection (socket, identity) the request arrived on. */
  connection: ConnectionInfo | null = null;
  /** Protocol revision negotiated by `initialize` on this connection. */
  protocolVersion: string | null = null;
  /** The session serving this request, for server-initiated notifications. */
  session: McpServerSession | null = null;

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

  /**
   * Send a notification to the client on this request's connection (e.g.
   * `notifications/progress`). Resolves `false` when there is no session or
   * the connection is closed.
   */
  notify(method: string, params?: unknown): Promise<boolean> {
    return this.session
      ? this.session.notify(method, params)
      : Promise.resolve(false);
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
 * MCP version negotiation: answer with the client's requested revision when
 * the server supports it, otherwise with the latest revision the server
 * supports (the client then decides whether it can continue).
 * @param requested `protocolVersion` from the client's `initialize`.
 * @param supported Revisions the server speaks, newest first.
 */
export function negotiateProtocolVersion(
  requested: string,
  supported: readonly string[] = SUPPORTED_PROTOCOL_VERSIONS,
): string {
  if (supported.includes(requested)) return requested;
  return supported[0] ?? LATEST_PROTOCOL_VERSION;
}

/** Options for {@link McpServer.serveConnection}. */
export interface ServeConnectionOptions {
  /**
   * Who is on the other end; copied to every `ctx.connection`. Default: a
   * fresh id with transport `"custom"`.
   */
  connection?: ConnectionInfo;
  /** Handler for this connection only (default: the server's handler). */
  handler?: McpHandler;
  /**
   * When the client's input ends (EOF), abort in-flight requests and drop
   * their responses instead of finishing them. Default `false` — a
   * half-closed client (`echo req | server`) still gets its answers. Socket
   * listeners served with {@link McpServer.serve} default to `true`.
   */
  abortOnEof?: boolean;
}

/** Options for {@link McpServer.serve}. */
export interface ServeOptions {
  /**
   * Build the handler for each accepted connection (e.g. one per terminal
   * tile). A throwing/rejecting factory closes that connection. Default: the
   * server's handler for every connection.
   */
  handlerFactory?: (info: ConnectionInfo) => McpHandler | Promise<McpHandler>;
  /** Called with each new session (track it to send notifications). */
  onSession?: (session: McpServerSession) => void;
  /** See {@link ServeConnectionOptions.abortOnEof}. Default `true`. */
  abortOnEof?: boolean;
  /**
   * Close the sessions this call started once the listener stops
   * (default `true`), so closing a listener tears its connections down.
   */
  closeSessionsOnStop?: boolean;
}

interface InFlight {
  key: string;
  controller: AbortController;
  /** Set when the client cancelled it or the connection is gone: no response. */
  dropped: boolean;
}

const idKey = (id: JsonRpcId | undefined): string => JSON.stringify(id ?? null);

/** A JSON-RPC notification has no id (or a null one) and is never answered. */
const isNotification = (request: JsonRpcRequest): boolean =>
  request.id === undefined || request.id === null;

/**
 * One MCP connection being served: reads requests from a
 * {@link ServerTransport}, runs the middleware chain on each in arrival order,
 * then dispatches requests **concurrently** (a slow `tools/call` never stalls
 * `tools/list` or another call) and writes each response — matched by `id` —
 * as soon as it is ready, one write at a time.
 *
 * Handles `ping`, `notifications/cancelled` (aborts the request's
 * `ctx.signal`, suppresses its response), `logging/setLevel`, and lets the
 * host push `notifications/tools/list_changed` / `notifications/message`.
 * Normally created by {@link McpServer.serveConnection} / {@link McpServer.serve}.
 */
export class McpServerSession {
  /** Who is on the other end of this connection. */
  readonly connection: ConnectionInfo;
  /** Connection-scoped metadata, shared by every request's `ctx.metadata`. */
  readonly metadata: Map<string, unknown> = new Map();
  /**
   * Resolves once the connection has ended: input closed, in-flight requests
   * settled, transport closed and the handler's `onShutdown` run.
   */
  readonly done: Promise<void>;

  #transport: ServerTransport;
  #handler: McpHandler;
  #middleware: MiddlewareChain;
  #abortOnEof: boolean;
  #clientInfo: ClientInfo | null = null;
  #protocolVersion: string | null = null;
  #initialized = false;
  #logLevel: LoggingLevel | null = null;
  #inflight = new Map<string, InFlight>();
  #tasks = new Set<Promise<void>>();
  #writeChain: Promise<unknown> = Promise.resolve();
  #writable = true;
  #closing = false;
  #transportClosed = false;

  /**
   * Start serving `transport` immediately (dispatch begins on the next
   * microtask).
   * @param transport The connection to serve.
   * @param handler The application handler for this connection.
   * @param middleware Middleware run on every request/response.
   * @param options Connection info and EOF behaviour.
   */
  constructor(
    transport: ServerTransport,
    handler: McpHandler,
    middleware: MiddlewareChain = new MiddlewareChain(),
    options: ServeConnectionOptions = {},
  ) {
    this.#transport = transport;
    this.#handler = handler;
    this.#middleware = middleware;
    this.#abortOnEof = options.abortOnEof ?? false;
    this.connection = options.connection ??
      { id: nextConnectionId("conn"), transport: "custom" };
    this.done = this.#run();
  }

  /** Client name/version from `initialize`, or `null` before it. */
  get clientInfo(): ClientInfo | null {
    return this.#clientInfo;
  }

  /** Negotiated protocol revision, or `null` before `initialize`. */
  get protocolVersion(): string | null {
    return this.#protocolVersion;
  }

  /** Whether `initialize` has completed on this connection. */
  get initialized(): boolean {
    return this.#initialized;
  }

  /** Requests currently being processed. */
  get inFlight(): number {
    return this.#inflight.size;
  }

  /** Whether the session is closing or closed (no more writes). */
  get closed(): boolean {
    return this.#closing || !this.#writable;
  }

  /**
   * Send a JSON-RPC notification to the client. Resolves `true` once written,
   * `false` if the connection is closed or the write failed.
   */
  notify(method: string, params?: unknown): Promise<boolean> {
    return this.#send(createJsonRpcNotification(method, params));
  }

  /**
   * Tell the client the tool list changed (`notifications/tools/list_changed`);
   * it will re-fetch `tools/list`. Declare `capabilities.tools.listChanged`.
   */
  notifyToolsListChanged(): Promise<boolean> {
    return this.notify("notifications/tools/list_changed");
  }

  /**
   * Send a log entry (`notifications/message`). Entries below the level the
   * client set with `logging/setLevel` are skipped (resolves `false`).
   * Declare `capabilities.logging`.
   * @param level Severity.
   * @param data Any JSON-serializable payload (string or object).
   * @param logger Optional logger name.
   */
  sendLogMessage(
    level: LoggingLevel,
    data: unknown,
    logger?: string,
  ): Promise<boolean> {
    if (
      this.#logLevel !== null &&
      LOG_LEVELS.indexOf(level) < LOG_LEVELS.indexOf(this.#logLevel)
    ) {
      return Promise.resolve(false);
    }
    return this.notify("notifications/message", {
      level,
      ...(logger !== undefined ? { logger } : {}),
      data,
    });
  }

  /**
   * Close the connection: abort in-flight requests (their responses are
   * dropped), close the transport and wait for {@link done}. A transport
   * without `close()` (stdio) ends only when its input does.
   */
  async close(): Promise<void> {
    if (!this.#closing) {
      this.#closing = true;
      this.#abortAll("session closed");
      await this.#closeTransport();
    }
    await this.done;
  }

  async #run(): Promise<void> {
    // Let the creator attach to the session before the first dispatch.
    await Promise.resolve();
    for (
      let line = await this.#read();
      line !== null;
      line = await this.#read()
    ) {
      try {
        await this.#onLine(line);
      } catch (e) {
        console.error(`MCP dispatch error: ${e}`);
      }
    }
    await this.#finish();
  }

  /** The next request line, or `null` once the session should stop reading. */
  async #read(): Promise<string | null> {
    if (this.#closing) return null;
    try {
      return await this.#transport.readRequest();
    } catch (e) {
      if (!this.#closing) console.error(`Transport read error: ${e}`);
      return null;
    }
  }

  /** Input ended: settle in-flight work, close the transport, run `onShutdown`. */
  async #finish(): Promise<void> {
    if (this.#abortOnEof || this.#closing) this.#abortAll("connection closed");
    while (this.#tasks.size > 0) await Promise.allSettled([...this.#tasks]);
    await this.#writeChain;
    this.#writable = false;
    await this.#closeTransport();
    try {
      await this.#handler.onShutdown?.(this.connection);
    } catch (e) {
      console.error(`Handler onShutdown failed: ${e}`);
    }
  }

  async #closeTransport(): Promise<void> {
    if (this.#transportClosed) return;
    this.#transportClosed = true;
    try {
      await this.#transport.close?.();
    } catch { /* already closed */ }
  }

  #abortAll(reason: string): void {
    for (const entry of this.#inflight.values()) {
      entry.dropped = true;
      entry.controller.abort(new DOMException(reason, "AbortError"));
    }
  }

  async #onLine(line: string): Promise<void> {
    const request = await this.#parse(line);
    if (request === null || await this.#lifecycle(request)) return;

    const controller = new AbortController();
    const ctx = this.#context(request.id ?? null, controller.signal);
    const middlewareError = await this.#middleware.processRequest(
      request,
      ctx,
    );
    // JSON-RPC: a notification (no id) is never answered, not even to reject it.
    if (isNotification(request)) return;
    if (middlewareError) {
      await this.#send({
        jsonrpc: "2.0",
        id: request.id,
        error: middlewareError,
      });
      return;
    }

    // `initialize` runs inline so every later request sees its client info.
    if (request.method === "initialize") {
      await this.#respond(request, ctx);
      return;
    }
    this.#spawn(request, ctx, controller);
  }

  /**
   * Decode one line into a request. Malformed input is answered here (parse
   * error / invalid request) and yields `null`, as does a response from the
   * client (the server sends no requests, so there is nothing to match).
   */
  async #parse(line: string): Promise<JsonRpcRequest | null> {
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch (e) {
      const error = AgentNetworkError.parseError(String(e));
      await this.#send({
        jsonrpc: "2.0",
        id: null,
        error: error.toJsonRpcError(),
      });
      return null;
    }
    if (
      typeof message !== "object" || message === null || Array.isArray(message)
    ) {
      await this.#send(invalidRequest(null));
      return null;
    }
    const request = message as JsonRpcRequest;
    if (typeof request.method === "string") return request;
    const isResponse = "result" in request || "error" in request;
    if (!isNotification(request) && !isResponse) {
      await this.#send(invalidRequest(request.id));
    }
    return null;
  }

  /**
   * Lifecycle messages answered before the middleware chain: a ping must
   * always work, and a cancellation only touches the client's own requests.
   * Returns whether `request` was one of them.
   */
  async #lifecycle(request: JsonRpcRequest): Promise<boolean> {
    if (request.method === "notifications/cancelled") {
      this.#cancel(request.params);
      return true;
    }
    if (request.method !== "ping") return false;
    if (!isNotification(request)) {
      await this.#send({ jsonrpc: "2.0", id: request.id, result: {} });
    }
    return true;
  }

  /** Dispatch, run response middleware and write the response (unless dropped). */
  async #respond(
    request: JsonRpcRequest,
    ctx: RequestContext,
    entry?: InFlight,
  ): Promise<void> {
    let response: JsonRpcResponse;
    try {
      response = await this.#dispatch(request, ctx);
    } finally {
      if (entry && this.#inflight.get(entry.key) === entry) {
        this.#inflight.delete(entry.key);
      }
    }
    // MCP: no response for a request the client cancelled.
    if (entry?.dropped) return;
    await this.#middleware.processResponse(response, ctx);
    await this.#send(response);
  }

  /** Run a request concurrently, tracked in-flight under its id until it settles. */
  #spawn(
    request: JsonRpcRequest,
    ctx: RequestContext,
    controller: AbortController,
  ): void {
    const key = idKey(request.id);
    const entry: InFlight = { key, controller, dropped: false };
    this.#inflight.set(key, entry);
    const task: Promise<void> = this.#respond(request, ctx, entry)
      .catch((e) => console.error(`MCP request ${key} failed: ${e}`))
      .finally(() => this.#tasks.delete(task));
    this.#tasks.add(task);
  }

  #cancel(params: unknown): void {
    const p = (typeof params === "object" && params !== null ? params : {}) as {
      requestId?: JsonRpcId;
      reason?: unknown;
    };
    if (p.requestId === undefined || p.requestId === null) return;
    const entry = this.#inflight.get(idKey(p.requestId));
    if (!entry) return; // already finished, or unknown: ignore per spec
    entry.dropped = true;
    entry.controller.abort(
      new DOMException(
        typeof p.reason === "string" ? p.reason : "cancelled by client",
        "AbortError",
      ),
    );
  }

  #context(requestId: JsonRpcId, signal: AbortSignal): RequestContext {
    const ctx = new RequestContext(requestId);
    ctx.clientInfo = this.#clientInfo;
    ctx.initialized = this.#initialized;
    ctx.metadata = this.metadata;
    ctx.signal = signal;
    ctx.connection = this.connection;
    ctx.protocolVersion = this.#protocolVersion;
    ctx.session = this;
    return ctx;
  }

  /** Serialize `message` and queue it behind every earlier write. */
  #send(message: object): Promise<boolean> {
    if (!this.#writable) return Promise.resolve(false);
    const json = JSON.stringify(message);
    const result = this.#writeChain.then(async () => {
      if (!this.#writable) return false;
      try {
        await this.#transport.writeResponse(json);
        return true;
      } catch (e) {
        this.#onWriteFailure(e);
        return false;
      }
    });
    this.#writeChain = result;
    return result;
  }

  #onWriteFailure(e: unknown): void {
    if (!this.#writable) return;
    this.#writable = false;
    if (!this.#closing) console.error(`Transport write error: ${e}`);
    // The peer is gone: stop work on its behalf and end the read loop.
    this.#abortAll("connection lost");
    this.#closeTransport();
  }

  /** Route a request by `method`; unknown methods get a `MethodNotFound` error response. */
  async #dispatch(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcResponse> {
    switch (request.method) {
      case "initialize":
        return await this.#initialize(request, ctx);
      case "tools/list":
        return this.#listTools(request, ctx);
      case "tools/call":
        return await this.#callTool(request, ctx);
      case "logging/setLevel":
        return this.#setLogLevel(request);
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
   * Handle `initialize`: record the client info, negotiate the protocol
   * revision, mark the connection initialized, invoke the handler's optional
   * `onInitialize` (failures are logged, not returned), and reply with the
   * negotiated revision plus the handler's capabilities and server info.
   */
  async #initialize(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcResponse> {
    const params = parseInitializeParams(request.params);
    const protocolVersion = negotiateProtocolVersion(params.protocolVersion);

    this.#clientInfo = {
      name: params.clientInfo.name,
      version: params.clientInfo.version,
    };
    this.#protocolVersion = protocolVersion;
    this.#initialized = true;
    ctx.clientInfo = this.#clientInfo;
    ctx.protocolVersion = protocolVersion;
    ctx.setInitialized();

    if (this.#handler.onInitialize) {
      try {
        await this.#handler.onInitialize(params, ctx);
      } catch (e) {
        console.error(`Handler onInitialize failed: ${e}`);
      }
    }

    const result: InitializeResult = {
      protocolVersion,
      capabilities: this.#handler.capabilities(),
      serverInfo: this.#handler.serverInfo(),
    };
    return { jsonrpc: "2.0", id: request.id, result };
  }

  /** Handle `tools/list`: reply with the handler's tools as `{ name, description, inputSchema }`. */
  #listTools(request: JsonRpcRequest, ctx: RequestContext): JsonRpcResponse {
    try {
      const tools = this.#handler.listTools(ctx).map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema,
      }));
      return { jsonrpc: "2.0", id: request.id, result: { tools } };
    } catch (e) {
      return {
        jsonrpc: "2.0",
        id: request.id,
        error: AgentNetworkError.internal(String(e)).toJsonRpcError(),
      };
    }
  }

  /**
   * Handle `tools/call`: reject with `InvalidParams` when `params` or
   * `params.name` is missing, otherwise call the handler with
   * `params.arguments` (default `{}`); a thrown error becomes an `Internal`
   * error response.
   */
  async #callTool(
    request: JsonRpcRequest,
    ctx: RequestContext,
  ): Promise<JsonRpcResponse> {
    const params = request.params as Record<string, unknown> | undefined;
    if (!params || typeof params !== "object") {
      const error = AgentNetworkError.invalidParams(
        "Missing params for tools/call",
      );
      return { jsonrpc: "2.0", id: request.id, error: error.toJsonRpcError() };
    }

    const toolName = params.name as string | undefined;
    if (!toolName) {
      const error = AgentNetworkError.invalidParams(
        "Missing 'name' in tools/call",
      );
      return { jsonrpc: "2.0", id: request.id, error: error.toJsonRpcError() };
    }

    const args = (params.arguments as Record<string, unknown>) ?? {};

    try {
      const result = await this.#handler.callTool(toolName, args, ctx);
      return { jsonrpc: "2.0", id: request.id, result };
    } catch (e) {
      const error = AgentNetworkError.internal(String(e));
      return { jsonrpc: "2.0", id: request.id, error: error.toJsonRpcError() };
    }
  }

  /** Handle `logging/setLevel`: remember the minimum level for {@link sendLogMessage}. */
  #setLogLevel(request: JsonRpcRequest): JsonRpcResponse {
    const level = (request.params as { level?: unknown } | undefined)?.level;
    if (!LOG_LEVELS.includes(level as LoggingLevel)) {
      const error = AgentNetworkError.invalidParams(
        `Unknown log level: ${String(level)}`,
      );
      return { jsonrpc: "2.0", id: request.id, error: error.toJsonRpcError() };
    }
    this.#logLevel = level as LoggingLevel;
    return { jsonrpc: "2.0", id: request.id, result: {} };
  }
}

function invalidRequest(id: JsonRpcId): JsonRpcResponse {
  return {
    jsonrpc: "2.0",
    id,
    error: { code: -32600, message: "Invalid Request" },
  };
}

/**
 * MCP server: an {@link McpHandler} plus a middleware chain, served over one
 * transport ({@link run}, stdio by default) or many connections
 * ({@link serve} on a socket listener, {@link serveConnection} per stream).
 * Each connection is an {@link McpServerSession} that handles `initialize`
 * (with protocol negotiation), `ping`, `tools/list`, `tools/call`,
 * `logging/setLevel` and `notifications/cancelled`, dispatching requests
 * concurrently and never answering notifications.
 * Equivalent to Rust `McpServer`.
 */
export class McpServer {
  private handler: McpHandler;
  private middleware: MiddlewareChain;
  private transport: ServerTransport | undefined;
  #sessions = new Set<McpServerSession>();

  /**
   * Create a server with an empty middleware chain.
   * @param handler The application handler.
   * @param transport Transport for {@link run} (default: stdio, opened only
   *   when `run()` is called).
   */
  constructor(handler: McpHandler, transport?: ServerTransport) {
    this.handler = handler;
    this.middleware = new MiddlewareChain();
    this.transport = transport;
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

  /** Sessions currently being served (all transports). */
  get sessions(): ReadonlySet<McpServerSession> {
    return this.#sessions;
  }

  /** Run the server on its transport (default stdio) until the transport closes. */
  async run(): Promise<void> {
    const transport = this.transport ?? new StdioServerTransport();
    this.transport = transport;
    console.error("MCP server starting");
    const kind = transport instanceof StdioServerTransport ? "stdio" : "custom";
    const session = this.serveConnection(transport, {
      connection: { id: nextConnectionId(kind), transport: kind },
    });
    await session.done;
    console.error("MCP server shut down");
  }

  /**
   * Serve one connection (any {@link ServerTransport}) and return its session
   * immediately; await `session.done` for its end. The connection info lands
   * on every request's `ctx.connection`.
   */
  serveConnection(
    transport: ServerTransport,
    options: ServeConnectionOptions = {},
  ): McpServerSession {
    const session = new McpServerSession(
      transport,
      options.handler ?? this.handler,
      this.middleware,
      options,
    );
    this.#sessions.add(session);
    session.done.finally(() => this.#sessions.delete(session));
    return session;
  }

  /**
   * Serve every connection a listener yields (e.g. a
   * `UnixSocketServerTransport`), one session each, until the listener is
   * closed. Resolves after the listener stops and (by default) its sessions
   * have been closed.
   */
  async serve(
    listener: AsyncIterable<AcceptedConnection>,
    options: ServeOptions = {},
  ): Promise<void> {
    const mine = new Set<McpServerSession>();
    const starting = new Set<Promise<void>>();
    const start = async ({ transport, info }: AcceptedConnection) => {
      let handler = this.handler;
      if (options.handlerFactory) {
        try {
          handler = await options.handlerFactory(info);
        } catch (e) {
          console.error(`handlerFactory failed for ${info.id}: ${e}`);
          await transport.close();
          return;
        }
      }
      const session = this.serveConnection(transport, {
        connection: info,
        handler,
        abortOnEof: options.abortOnEof ?? true,
      });
      mine.add(session);
      session.done.finally(() => mine.delete(session));
      options.onSession?.(session);
    };

    for await (const accepted of listener) {
      const p: Promise<void> = start(accepted).finally(() =>
        starting.delete(p)
      );
      starting.add(p);
    }
    await Promise.allSettled([...starting]);
    if (options.closeSessionsOnStop ?? true) {
      await Promise.allSettled([...mine].map((s) => s.close()));
    }
  }
}
