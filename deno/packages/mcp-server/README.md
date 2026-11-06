# @rullama/mcp-server

MCP-compliant tool server framework. Extracted from `@rullama/network` in
v0.11.0 to mirror Rust's standalone `rullama-mcp-server` crate.

What it implements:

- **`McpServer`** -- middleware pipeline + one `McpServerSession` per connection
  (`run()` on stdio, `serve(listener)` on sockets, `serveConnection(transport)`
  for any stream). A session handles `initialize` (protocol-version
  negotiation), `ping`, `tools/list`, `tools/call`, `logging/setLevel` and
  `notifications/cancelled`, dispatching requests concurrently; every other
  method (resources, prompts, sampling, …) gets a method-not-found error.
- **`McpHandler`** -- the interface your application implements (`serverInfo`,
  `capabilities`, `listTools`, `callTool`, optional `onInitialize` /
  `onShutdown`). `RequestContext` carries the per-request `signal`, the
  `connection` (id, socket path, identity) and the negotiated `protocolVersion`.
- **`McpToolRegistry`** -- tool registration and dispatch by name.
- **`MiddlewareChain`** with `AuthMiddleware` (bearer token, constant-time
  compare), `LoggingMiddleware`, `RateLimitMiddleware` (per-second, optional
  per-tool limits) and `ToolFilterMiddleware` (allow / deny lists).
- **Transports** -- `StdioServerTransport` (stdin/stdout),
  `UnixSocketServerTransport` (listener on a socket path) and
  `TcpServerTransport` (listener on a host/port, with an optional first-line
  pre-auth hook), all newline-delimited JSON-RPC with a 16 MiB line cap.
  Implement `ServerTransport` for anything else.

The MCP _client_ is `@rullama/mcp-client`; this package imports its wire types.

## Install

```sh
deno add jsr:@rullama/mcp-server
```

## Quick Example

```ts
import type { CallToolResult } from "@rullama/mcp-client";
import {
  AuthMiddleware,
  LoggingMiddleware,
  type McpHandler,
  McpServer,
  McpToolRegistry,
  RateLimitMiddleware,
  type RequestContext,
  ToolFilterMiddleware,
} from "@rullama/mcp-server";

const registry = new McpToolRegistry();
registry.register(
  "echo",
  "Echo the message back",
  {
    type: "object",
    properties: { message: { type: "string" } },
    required: ["message"],
  },
  (
    args: Record<string, unknown>,
    _ctx: RequestContext,
  ): Promise<CallToolResult> =>
    Promise.resolve({
      content: [{ type: "text", text: `Echo: ${args.message}` }],
    }),
);

const handler: McpHandler = {
  serverInfo: () => ({ name: "echo-server", version: "1.0.0" }),
  capabilities: () => ({ tools: {} }),
  listTools: () => registry.listTools(),
  callTool: (name, args, ctx) => registry.dispatch(name, args, ctx),
};

const server = new McpServer(handler) // stdio transport by default
  .withMiddleware(new LoggingMiddleware())
  .withMiddleware(new AuthMiddleware(Deno.env.get("MCP_TOKEN") ?? "dev"))
  .withMiddleware(new RateLimitMiddleware(20).withToolLimit("echo", 5))
  .withMiddleware(ToolFilterMiddleware.allowOnly(["echo"]));

await server.run(); // until stdin closes
```

Run it with `deno run --allow-env server.ts` and point any MCP client at the
process; see `examples/network/mcp_server.ts` for a middleware walkthrough
without a live transport.

## Serving over unix sockets (one connection per client)

`UnixSocketServerTransport.listen(path)` binds a socket file and yields one
`SocketServerTransport` per accepted connection; `McpServer.serve(listener)`
runs an `McpServerSession` on each. Framing is the same newline-delimited
JSON-RPC as stdio, so a stdio-only MCP client can reach the socket through
`nc -U <path>`. Every request's `ctx.connection` says which connection (and
which `identity`) it came from.

```ts
import type { CallToolResult } from "@rullama/mcp-client";
import {
  type McpHandler,
  McpServer,
  type McpServerSession,
  UnixSocketServerTransport,
} from "@rullama/mcp-server";

function handlerFor(tileId: string): McpHandler {
  return {
    serverInfo: () => ({ name: "hub", version: "1.0.0" }),
    capabilities: () => ({ tools: { listChanged: true }, logging: {} }),
    listTools: () => [{
      name: "whoami",
      description: "Which tile am I attached to?",
      inputSchema: { type: "object" },
    }],
    callTool: (_name, _args, ctx): Promise<CallToolResult> => {
      // ctx.connection.identity is the listener's `identity` (or what a
      // pre-auth hook returned); ctx.signal aborts on notifications/cancelled
      // or client disconnect.
      const { tile } = ctx.connection!.identity as { tile: string };
      return Promise.resolve({
        content: [{ type: "text", text: `tile ${tile} (${tileId})` }],
      });
    },
  };
}

const server = new McpServer(handlerFor("default"));
const sessions = new Set<McpServerSession>();

// One socket per tile. A stale file from a crashed run is replaced; the file
// is chmod 0600 and removed again on close().
const listener = await UnixSocketServerTransport.listen(
  "/tmp/hub-tile-1.sock",
  {
    identity: { tile: "1" },
  },
);
const serving = server.serve(listener, {
  handlerFactory: (info) =>
    handlerFor((info.identity as { tile: string }).tile),
  onSession: (s) => {
    sessions.add(s);
    s.done.then(() => sessions.delete(s));
  },
});

// Later: push notifications to the connected client(s)…
for (const s of sessions) await s.notifyToolsListChanged();
// …and tear the tile down (closes its sessions, removes the socket file).
listener.close();
await serving;
```

Claude Code / Codex config for that tile:
`{"command": "nc", "args": ["-U",
"/tmp/hub-tile-1.sock"]}`.

`TcpServerTransport.listen({ port: 0, preAuth })` is the loopback-TCP
equivalent. Any local process can reach a loopback port, so pass a `preAuth`
hook: it receives the first line the client sends (before any JSON-RPC) and
returns `{ ok: true, identity }` or `{ ok: false, reason }`; a rejected client
gets one `-32003 Unauthorized` line and is disconnected. The
`@rullama/mcp-client` side is
`SocketTransport.connectTcp(host, port, { preAuthLine: token })`.

### Per-connection behaviour

- Requests run **concurrently**: a 60 s `tools/call` does not stall `tools/list`
  or another call. Middleware still runs in arrival order; responses are written
  one at a time as each finishes, matched by `id`.
- `notifications/cancelled` aborts that request's `ctx.signal` and its response
  is not sent. `ping` is answered (before middleware, so it works pre-auth).
  `logging/setLevel` sets the floor for `session.sendLogMessage`.
- `initialize` echoes the client's `protocolVersion` when supported
  (`2025-06-18`, `2025-03-26`, `2024-11-05` — `SUPPORTED_PROTOCOL_VERSIONS`,
  shared with `@rullama/mcp-client`), otherwise answers with the latest.
- Client EOF: with `serve()` (default `abortOnEof: true`) in-flight requests are
  aborted; with `run()` / `serveConnection()` (default `false`) they finish and
  their responses are still written. Either way the transport is closed and the
  handler's `onShutdown(connection)` runs once per connection.
- `session.notify(method, params)`, `notifyToolsListChanged()`,
  `sendLogMessage(level, data, logger?)` and `ctx.notify(...)` send
  server-initiated notifications; they resolve `false` once the connection is
  gone.
