# @rullama/network

Agent networking layer for the rullama. Provides an MCP server framework with
middleware, agent identity, message routing, peer discovery, and client
connectivity.

Equivalent to the Rust `rullama-network` crate.

> **Transport security.** `@rullama/network` is an _unauthenticated_ transport
> layer: `MessageEnvelope.sender` is a self-asserted string and envelopes carry
> no signature, nonce or timestamp check. Use it only between processes you
> already trust (one host, one operator), or wrap it in an authenticated channel
> (mTLS, an SSH tunnel, an authenticated WebSocket). Signed envelopes with
> replay protection are planned for 0.13.

## Install

```sh
deno add @rullama/network
```

## Quick Example

```ts
import {
  LoggingMiddleware,
  McpServer,
  McpToolRegistry,
  RateLimitMiddleware,
} from "@rullama/network";
import type { McpHandler } from "@rullama/network";

// Define a handler
const handler: McpHandler = {
  name: () => "my-agent",
  version: () => "1.0.0",
  tools: () => toolRegistry.definitions(),
  callTool: async (name, args) => {
    return { content: [{ type: "text", text: `Executed ${name}` }] };
  },
};

// Build tool definitions
const toolRegistry = new McpToolRegistry();
toolRegistry.register({
  name: "greet",
  description: "Say hello",
  inputSchema: { type: "object", properties: { name: { type: "string" } } },
  handler: async (args) => ({
    content: [{ type: "text", text: `Hello, ${args.name}!` }],
  }),
});

// Create server with middleware
const server = new McpServer(handler)
  .withMiddleware(new LoggingMiddleware())
  .withMiddleware(new RateLimitMiddleware(100, 60_000));

// Run the server (reads/writes JSON-RPC over stdio)
await server.run();
```

## Key Exports

| Export                                               | Description                                               |
| ---------------------------------------------------- | --------------------------------------------------------- |
| `McpServer`                                          | MCP server with JSON-RPC dispatch and middleware pipeline |
| `McpToolRegistry`                                    | Tool definition container with handler dispatch           |
| `MiddlewareChain`                                    | Ordered middleware pipeline                               |
| `AuthMiddleware`                                     | Token-based authentication middleware                     |
| `LoggingMiddleware`                                  | Request/response logging                                  |
| `RateLimitMiddleware`                                | Rate limiting per client                                  |
| `ToolFilterMiddleware`                               | Allow/deny list for tool access                           |
| `DirectRouter` / `BroadcastRouter` / `ContentRouter` | Message routing strategies                                |
| `ManualDiscovery`                                    | Static peer discovery                                     |
| `AgentNetworkClient`                                 | Client for connecting to agent network nodes              |
