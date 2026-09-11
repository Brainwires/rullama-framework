# @rullama/mcp-client

Model Context Protocol (MCP) client for the rullama. Connect to external MCP
servers, discover and call tools, read resources, and fetch prompts.

The client offers protocol version `2025-06-18` when it initializes and accepts
servers that answer with `2025-06-18`, `2025-03-26` or `2024-11-05`
(`LATEST_PROTOCOL_VERSION` / `SUPPORTED_PROTOCOL_VERSIONS`). Only the **stdio**
transport is implemented: each server is spawned as a subprocess and spoken to
over newline-delimited JSON-RPC on its stdin/stdout. There is no HTTP, SSE or
WebSocket transport yet.

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
| `McpClient`                                               | Client that manages stdio connections to one or more MCP servers    |
| `LATEST_PROTOCOL_VERSION` / `SUPPORTED_PROTOCOL_VERSIONS` | Protocol version offered on initialize and the versions accepted    |
| `McpConfigManager` / `McpServerConfig`                    | Load, add, remove and save server configurations on disk            |
| `StdioTransport`                                          | Subprocess transport speaking newline-delimited JSON-RPC            |
| `Transport`                                               | Thin wrapper around a `StdioTransport` (the only transport today)   |
| JSON-RPC types                                            | `JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcNotification`, helpers |
| MCP types                                                 | `McpTool`, `McpResource`, `McpPrompt`, `CallToolResult`, etc.       |
