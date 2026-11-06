# @rullama/mcp-client

Model Context Protocol (MCP) client for the rullama. Connect to external MCP
servers, discover and call tools, read resources, and fetch prompts.

The client offers protocol version `2025-06-18` when it initializes and accepts
servers that answer with `2025-06-18`, `2025-03-26` or `2024-11-05`
(`LATEST_PROTOCOL_VERSION` / `SUPPORTED_PROTOCOL_VERSIONS`). Transports:
**stdio** (`connect(config)` spawns the server as a subprocess) and **unix
socket / TCP**
(`connectTransport(name, await SocketTransport.connectUnix(path))`), both
newline-delimited JSON-RPC. There is no HTTP, SSE or WebSocket transport yet.

Each connection has a background reader that matches responses to requests by
`id`, so calls can run concurrently on one connection and complete in any order.
Server notifications go to `onNotification(handler)` subscribers (and to a
call's own `onNotification`); a server `ping` is answered automatically. Every
request method takes an optional `{ signal }`: aborting it rejects the promise
and sends `notifications/cancelled`.

```ts
import { McpClient, SocketTransport } from "@rullama/mcp-client";

const client = new McpClient("my-agent", "1.0.0");
await client.connectTransport(
  "hub",
  await SocketTransport.connectUnix("/tmp/hub-tile-1.sock"),
);
client.onNotification((server, n) => console.log(server, n.method));

const ac = new AbortController();
const slow = client.callTool("hub", "wait_for_message", {}, {
  signal: ac.signal,
});
console.log(await client.listTools("hub")); // not blocked by `slow`
ac.abort(); // rejects `slow`, server receives notifications/cancelled
await slow.catch(() => {});
await client.disconnect("hub");
```

Equivalent to the Rust `rullama-mcp` crate.

## Install

```sh
deno add @rullama/mcp-client
```

## Quick Example

```ts
import { McpClient } from "@rullama/mcp-client";

// Create a client and connect to an MCP server (spawned as a subprocess)
const client = McpClient.createDefault();

await client.connect({
  name: "filesystem",
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
});

// List available tools
const tools = await client.listTools("filesystem");
for (const tool of tools) {
  console.log(`${tool.name}: ${tool.description ?? ""}`);
}

// Call a tool
const result = await client.callTool("filesystem", "list_directory", {
  path: "/tmp",
});
for (const item of result.content) {
  if (item.type === "text") console.log(item.text);
}

// Disconnect when done
await client.disconnect("filesystem");
```

`connect()` takes a full `McpServerConfig` (`name`, `command`, `args`, optional
`env`); the `name` is the handle every later call uses. `McpConfigManager` loads
and saves the same config shape from `~/.rullama/mcp-config.json`.

## Key Exports

| Export                                                    | Description                                                         |
| --------------------------------------------------------- | ------------------------------------------------------------------- |
| `McpClient`                                               | Client managing concurrent connections to one or more MCP servers   |
| `LATEST_PROTOCOL_VERSION` / `SUPPORTED_PROTOCOL_VERSIONS` | Protocol version offered on initialize and the versions accepted    |
| `McpConfigManager` / `McpServerConfig`                    | Load, add, remove and save server configurations on disk            |
| `StdioTransport`                                          | Subprocess transport speaking newline-delimited JSON-RPC            |
| `SocketTransport`                                         | Unix-socket / TCP transport (`connectUnix`, `connectTcp`)           |
| `ClientTransport` / `Transport`                           | Transport interface / thin wrapper around any `ClientTransport`     |
| JSON-RPC types                                            | `JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcNotification`, helpers |
| MCP types                                                 | `McpTool`, `McpResource`, `McpPrompt`, `CallToolResult`, etc.       |
