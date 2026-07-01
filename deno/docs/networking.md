# Networking (Agent Network)

The `@rullama/network` package provides an MCP server framework, middleware
pipeline, agent identity, peer discovery, message routing, and a remote bridge.

## MCP Server Framework

Build MCP-compliant tool servers with `McpServer` and `McpToolRegistry`:

```ts
import {
  McpServer,
  McpToolRegistry,
  StdioServerTransport,
} from "@rullama/network";

const toolRegistry = new McpToolRegistry();

toolRegistry.register({
  name: "greet",
  description: "Greet someone",
  inputSchema: { type: "object", properties: { name: { type: "string" } } },
  handler: async (params) => ({
    content: [{ type: "text", text: `Hello, ${params.name}!` }],
  }),
});

const server = new McpServer(
  { name: "my-server", version: "1.0.0" },
  toolRegistry,
);
const transport = new StdioServerTransport();
await server.start(transport);
```

See: `../examples/agent-network/mcp_server.ts`.

## Middleware

The `MiddlewareChain` processes requests before they reach tool handlers.
Built-in middleware:

| Middleware             | Purpose                        |
| ---------------------- | ------------------------------ |
| `AuthMiddleware`       | API key / token authentication |
| `LoggingMiddleware`    | Request/response logging       |
| `RateLimitMiddleware`  | Per-client request throttling  |
| `ToolFilterMiddleware` | Allow/deny list for tools      |

```ts
import {
  AuthMiddleware,
  LoggingMiddleware,
  MiddlewareChain,
  RateLimitMiddleware,
} from "@rullama/network";

const chain = new MiddlewareChain();
chain.use(new LoggingMiddleware());
chain.use(new AuthMiddleware({ apiKeys: ["secret-key"] }));
chain.use(new RateLimitMiddleware({ maxRequestsPerMinute: 100 }));
```

Custom middleware implements the `Middleware` interface and returns
`middlewareContinue()` or `middlewareReject(reason)`.

## Agent Identity

Every agent has an `AgentIdentity` with capabilities and protocol support:

```ts
import { createAgentIdentity, defaultAgentCard } from "@rullama/network";

const identity = createAgentIdentity("agent-1", defaultAgentCard("agent-1"));
```

Types: `AgentIdentity`, `AgentCard`, `ProtocolId`.

## Message Routing

Three routing strategies for inter-agent messages:

| Router            | Strategy                               |
| ----------------- | -------------------------------------- |
| `DirectRouter`    | Point-to-point delivery                |
| `BroadcastRouter` | Deliver to all known peers             |
| `ContentRouter`   | Route based on message content / topic |

Messages are wrapped in `MessageEnvelope` with target, TTL, and correlation
metadata. Helpers: `directEnvelope`, `broadcastEnvelope`, `topicEnvelope`,
`replyEnvelope`.

## Peer Discovery

Use `ManualDiscovery` to register peers, or implement the `Discovery` interface
for custom protocols (mDNS, service registry, etc.).

```ts
import { ManualDiscovery, PeerTable } from "@rullama/network";

const discovery = new ManualDiscovery();
const peerTable = new PeerTable();
```

See: `../examples/agent-network/peer_discovery.ts`.

## Remote Bridge

The remote bridge enables cross-process agent communication with protocol
negotiation, heartbeats, and command queuing:

```ts
import {
  defaultBridgeConfig,
  RemoteBridge,
  RemoteBridgeManager,
} from "@rullama/network";

const bridge = new RemoteBridge(defaultBridgeConfig());
const manager = new RemoteBridgeManager();
```

Key types: `BridgeConfig`, `BridgeState`, `CommandQueue`, `HeartbeatCollector`,
`ProtocolMetrics`.

See: `../examples/agent-network/network_manager.ts`.

## Further Reading

- [A2A Protocol](./a2a.md) for the Google A2A protocol layer built on top of
  networking
- [Extensibility](./extensibility.md) for custom middleware and discovery
