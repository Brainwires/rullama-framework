# Networking

Two packages cover inter-agent networking since v0.11.0:

- `@rullama/mcp-server` -- the MCP tool-server framework (`McpServer`,
  `McpToolRegistry`, the middleware chain, the stdio transport). It implements
  four MCP methods: `initialize`, `notifications/initialized`, `tools/list` and
  `tools/call`. Resources, prompts and sampling are not served.
- `@rullama/network` -- agent identity and capability cards, message envelopes,
  routing strategies, a peer table, peer discovery, the remote relay bridge and
  a stdio relay client. It re-exports `AgentNetworkError` / `ErrorCode` from
  `@rullama/mcp-server`.

The MCP _client_ is `@rullama/mcp-client` (see
[getting-started.md](./getting-started.md#5-connect-to-an-mcp-server-optional)).

## MCP Server Framework

Register tools in an `McpToolRegistry`, implement `McpHandler` around it, and
run the server on stdio:

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
  "greet",
  "Greet someone",
  { type: "object", properties: { name: { type: "string" } } },
  (
    args: Record<string, unknown>,
    _ctx: RequestContext,
  ): Promise<CallToolResult> =>
    Promise.resolve({
      content: [{ type: "text", text: `Hello, ${args.name}!` }],
    }),
);

const handler: McpHandler = {
  serverInfo: () => ({ name: "my-server", version: "1.0.0" }),
  capabilities: () => ({ tools: {} }),
  listTools: () => registry.listTools(),
  callTool: (name, args, ctx) => registry.dispatch(name, args, ctx),
};

const server = new McpServer(handler) // stdio transport by default
  .withMiddleware(new LoggingMiddleware())
  .withMiddleware(new AuthMiddleware("secret-token"))
  .withMiddleware(new RateLimitMiddleware(10)) // requests per second
  .withMiddleware(ToolFilterMiddleware.deny(["rm_rf"]));

await server.run(); // until the transport closes
```

Pass a custom `ServerTransport` (`readRequest()` / `writeResponse()`) to the
constructor or `withTransport()` to embed the server elsewhere; the shipped
transport is `StdioServerTransport` only.

See: `../examples/network/mcp_server.ts`.

## Middleware

`MiddlewareChain` runs every layer's `processRequest(request, ctx)` before the
handler and the optional `processResponse` after it. Built-in middleware:

| Middleware             | Constructor                                                                 | Purpose                                    |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------------------------ |
| `AuthMiddleware`       | `new AuthMiddleware(token)`                                                 | Bearer token check (constant-time compare) |
| `LoggingMiddleware`    | `new LoggingMiddleware()`                                                   | Request/response logging to stderr         |
| `RateLimitMiddleware`  | `new RateLimitMiddleware(maxRequestsPerSecond)` + `.withToolLimit(name, n)` | Per-client / per-tool throttling           |
| `ToolFilterMiddleware` | `ToolFilterMiddleware.allowOnly([...])` / `.deny([...])`                    | Allow/deny list for `tools/call`           |

Custom middleware implements the `Middleware` interface and returns
`middlewareContinue()` or `middlewareReject(jsonRpcError)`.

## Agent Identity

```ts
import { createAgentIdentity, defaultAgentCard } from "@rullama/network";

const identity = createAgentIdentity("agent-1"); // random id + default card
const card = defaultAgentCard();
```

Types: `AgentIdentity`, `AgentCard`, `ProtocolId`, `TransportAddress`.

## Message Routing

Three routing strategies for inter-agent messages:

| Router            | Strategy                               |
| ----------------- | -------------------------------------- |
| `DirectRouter`    | Point-to-point delivery                |
| `BroadcastRouter` | Deliver to all known peers             |
| `ContentRouter`   | Route based on message content / topic |

Messages are wrapped in a `MessageEnvelope` with target, TTL and correlation
metadata. Helpers: `directEnvelope`, `broadcastEnvelope`, `topicEnvelope`,
`replyEnvelope`, `withTtl`, `withCorrelation`; payloads via `textPayload`,
`jsonPayload`, `binaryPayload`.

## Peer Discovery

Use `ManualDiscovery` to register peers, or implement the `Discovery` interface
for custom protocols (mDNS, service registry, etc.). `PeerTable` tracks known
peers and topic subscriptions for the routers.

```ts
import { ManualDiscovery, PeerTable } from "@rullama/network";

const discovery = new ManualDiscovery();
const peerTable = new PeerTable();
```

See: `../examples/network/peer_discovery.ts`,
`../examples/network/network_manager.ts`.

## Remote Bridge and Relay Client

`RemoteBridge` (with `defaultBridgeConfig()`, `RemoteBridgeManager`,
`CommandQueue`, `HeartbeatCollector`, protocol negotiation) connects an agent
process to a remote backend over authenticated HTTP polling;
`AgentNetworkClient` is a stdio relay client that talks to a subprocess.

## Security: an unauthenticated transport

`@rullama/network`'s messaging layer is an **unauthenticated transport**.
Envelopes are not signed or verified, `PeerTable` / `ManualDiscovery` trust
whatever is registered, and the routers deliver to any peer they know about.
Treat it as a local or trusted-network fabric and put authentication at the
edges: `AuthMiddleware` on an MCP server, the remote bridge's API key, or A2A
security schemes (`@rullama/a2a`). Do not expose the relay to untrusted peers.

## Further Reading

- [A2A Protocol](./a2a.md) for the Google A2A client
- [Extensibility](./extensibility.md) for custom middleware and discovery
