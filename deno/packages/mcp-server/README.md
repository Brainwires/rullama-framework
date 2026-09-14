# @rullama/mcp-server

MCP-compliant tool server framework. Extracted from `@rullama/network` in
v0.11.0 to mirror Rust's standalone `rullama-mcp-server` crate.

What it implements:

- **`McpServer`** -- request loop + middleware pipeline. It handles exactly four
  MCP methods: `initialize`, `notifications/initialized`, `tools/list` and
  `tools/call`; every other method (resources, prompts, sampling, …) gets a
  method-not-found error.
- **`McpHandler`** -- the interface your application implements (`serverInfo`,
  `capabilities`, `listTools`, `callTool`, optional `onInitialize` /
  `onShutdown`).
- **`McpToolRegistry`** -- tool registration and dispatch by name.
- **`MiddlewareChain`** with `AuthMiddleware` (bearer token, constant-time
  compare), `LoggingMiddleware`, `RateLimitMiddleware` (per-second, optional
  per-tool limits) and `ToolFilterMiddleware` (allow / deny lists).
- **`StdioServerTransport`** -- the only shipped transport (newline-delimited
  JSON-RPC over stdin/stdout, 16 MiB line cap). Implement `ServerTransport` for
  anything else.

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
