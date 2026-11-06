# @rullama/mcp-server

MCP-compliant tool server framework. Extracted from `@rullama/network` in
v0.11.0 to mirror Rust's standalone `rullama-mcp-server` crate.

What it implements:

- **`McpServer`** -- middleware pipeline + one `McpServerSession` per connection
  (`run()` on stdio, `serve(listener)` on sockets, `serveConnection(transport)`
  for any stream). A session handles `initialize` (protocol-version
  negotiation), `ping`, `tools/list`, `tools/call`, `logging/setLevel` and
  `notifications/cancelled`, dispatching requests concurrently; every other
  method (resources, prompts, sampling, `server/discover`, …) gets a
  method-not-found error.
- **`McpHandler`** -- the interface your application implements (`serverInfo`,
  `capabilities`, `listTools`, `callTool`, optional `onInitialize` /
  `onShutdown`). `RequestContext` carries the per-request `signal`, the
  `connection` (id, socket path, identity) and the negotiated `protocolVersion`.
- **`McpToolRegistry`** -- tool registration and dispatch by name.
- **`MiddlewareChain`** with `AuthMiddleware` (bearer token, constant-time
  compare), `LoggingMiddleware`, `RateLimitMiddleware` (per-second, optional
  per-tool limits), `ToolFilterMiddleware` (allow / deny lists) and
  `IdentityToolFilterMiddleware` (per-connection-identity tool visibility and
  call rejection).
- **Transports** -- `StdioServerTransport` (stdin/stdout),
  `UnixSocketServerTransport` (listener on a socket path) and
  `TcpServerTransport` (listener on a host/port, with an optional first-line
  pre-auth hook), all newline-delimited JSON-RPC with a 16 MiB line cap.
  `StreamServerTransport` / `StreamServerListener` wrap sockets the host opens
  itself (FFI, a pre-opened `Deno.Conn`, pipes) — see
  [Embedding with your own sockets](#embedding-with-your-own-sockets-least-privilege).
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
  // Optional Tool fields, emitted in tools/list.
  { annotations: { readOnlyHint: true, openWorldHint: false } },
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
  (`2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05` —
  `SUPPORTED_PROTOCOL_VERSIONS`, shared with `@rullama/mcp-client`), otherwise
  answers with the latest (`2025-11-25`).
- On a `2025-11-25` connection a tool that throws is reported as a tool
  execution error —
  `{ content: [{ type: "text", text: message }], isError:
  true }` — so the
  model can correct itself (SEP-1303); an `AgentNetworkError` for an unknown
  tool, a refusal or a rate limit stays a JSON-RPC error with its own code.
  Older revisions keep the `-32603` error. `tools/list` passes the optional
  `title`, `outputSchema`, `annotations`, `icons` and `_meta` fields of a tool
  through.
- `server/discover` (the stateless `2026-07-28` revision's probe, sent first by
  e.g. Claude Code 2.1.x) is answered immediately with `-32601` Method not
  found, before middleware. This server speaks the `initialize`-based revisions,
  and the spec's stdio fallback rule is that a dual-era client treats any
  non-modern error to that probe as "legacy server" and continues with
  `initialize`.
- Client EOF: with `serve()` (default `abortOnEof: true`) in-flight requests are
  aborted; with `run()` / `serveConnection()` (default `false`) they finish and
  their responses are still written. Either way the transport is closed and the
  handler's `onShutdown(connection)` runs once per connection.
- `session.notify(method, params)`, `notifyToolsListChanged()`,
  `sendLogMessage(level, data, logger?)` and `ctx.notify(...)` send
  server-initiated notifications; they resolve `false` once the connection is
  gone.

## Embedding with your own sockets (least privilege)

`UnixSocketServerTransport.listen` binds with `Deno.listen` and stats, chmods
and removes the socket file, so it needs `--allow-read` and `--allow-write` on
the path. A host that runs with fewer permissions — e.g. a packaged desktop app
with FFI but no `--allow-write`, `--allow-run` or `--allow-sys` — can open its
sockets another way and still get the same per-connection session handling
(framing, concurrent dispatch, cancellation, pre-auth, identity, `onShutdown`).
Nothing below touches the file system or the network.

- `StreamServerTransport.from({ readable, write, close? })` — one connection.
  `readable` is a `ReadableStream<Uint8Array>` or any
  `AsyncIterable<Uint8Array>`; `write(bytes)` may resolve to a byte count, and a
  short count is continued like `write(2)` (so `conn.write.bind(conn)` works
  as-is). Serve it with `server.serveConnection(transport, options)`.
- `new StreamServerListener(source, { transport, identity, preAuth })` — adapts
  your accept loop (`AsyncIterable<{ readable, write, close?, info? }>`) into a
  listener for `server.serve(...)`. `McpServer.serve` accepts any
  `AsyncIterable<AcceptedConnection<ServerTransport>>`, so you can also build
  the `{ transport, info }` pairs yourself.

```ts ignore
import {
  IdentityToolFilterMiddleware,
  McpServer,
  StreamServerListener,
} from "@rullama/mcp-server";

// Your FFI layer: socket()/bind()/listen() once, then accept() per client.
declare function ffiAccept(listenFd: number): Promise<number>;
declare function ffiRead(fd: number): Promise<Uint8Array | null>; // null = EOF
declare function ffiWrite(fd: number, bytes: Uint8Array): Promise<number>;
declare function ffiClose(fd: number): void;
declare const listenFd: number;

async function* fdChunks(fd: number) {
  for (let c = await ffiRead(fd); c !== null; c = await ffiRead(fd)) yield c;
}

async function* accepted() {
  while (true) {
    const fd = await ffiAccept(listenFd);
    yield {
      readable: fdChunks(fd),
      write: (bytes: Uint8Array) => ffiWrite(fd, bytes), // partial counts ok
      close: () => ffiClose(fd),
      info: { transport: "unix", path: "/run/app/tile-1.sock" },
    };
  }
}

const server = new McpServer(handler).withMiddleware(
  // Hide/refuse admin tools unless the connection's identity allows them.
  new IdentityToolFilterMiddleware((tool, identity) =>
    !tool.startsWith("admin_") ||
    (identity as { admin?: boolean } | undefined)?.admin === true
  ),
);

const listener = new StreamServerListener(accepted(), {
  identity: { tile: "1", admin: false },
});
await server.serve(listener, {
  // A non-MCP one-shot line protocol on the same socket.
  firstLine: (line) =>
    line.startsWith('{"app":"hook"')
      ? { action: "close", reply: JSON.stringify({ ok: true }) }
      : { action: "mcp" },
});
```

### First-line hooks on any connection

`serveConnection(transport, { preAuth })` runs the same first-line pre-auth as
the socket listeners (line consumed; `{ ok: false }` → one `-32003
Unauthorized`
line and close; `identity` → `ctx.connection.identity`).
`serveConnection(transport, { firstLine })` (also
`serve(listener, {
firstLine })`) instead sees the first line and returns
`{ action: "mcp", line?, identity? }` (hand it to MCP),
`{ action: "consume",
identity? }` (drop it, MCP starts with the next line) or
`{ action: "close",
reply? }` (write the reply line(s), close without MCP).
`firstLineTimeoutMs` bounds the wait (default 10 s for `preAuth`, none for
`firstLine`).

### Per-identity tool filtering

`new IdentityToolFilterMiddleware((tool, identity, ctx) => boolean)` decides per
request from `ctx.connection.identity`: tools the caller may not use are removed
from its `tools/list` result and its `tools/call` for them is rejected with
`-32001`.
`IdentityToolFilterMiddleware.allowFor(identity => names |
"*" | undefined)` is
the allow-list shorthand. A throwing predicate fails closed.
