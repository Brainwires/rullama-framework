# @rullama/mcp

Model Context Protocol (MCP) client for the Brainwires Agent Framework. Connect
to external MCP servers, discover and call tools, read resources, and fetch
prompts.

Equivalent to the Rust `rullama-mcp` crate.

## Install

```sh
deno add @rullama/mcp
```

## Quick Example

```ts
import { McpClient, McpConfigManager } from "@rullama/mcp-client";

// Create a client and connect to an MCP server
const client = McpClient.createDefault();

await client.connect("filesystem", {
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
});

// List available tools
const tools = await client.listTools("filesystem");
for (const tool of tools) {
  console.log(`${tool.name}: ${tool.description}`);
}

// Call a tool
const result = await client.callTool("filesystem", "list_directory", {
  path: "/tmp",
});
console.log(result);

// Disconnect when done
await client.disconnect("filesystem");
```

## Key Exports

| Export             | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `McpClient`        | Client that manages connections to MCP servers                |
| `McpConfigManager` | Load and manage MCP server configurations                     |
| `StdioTransport`   | Stdio-based transport for subprocess communication            |
| `Transport`        | Base transport class                                          |
| JSON-RPC types     | `JsonRpcRequest`, `JsonRpcResponse`, helpers                  |
| MCP types          | `McpTool`, `McpResource`, `McpPrompt`, `CallToolResult`, etc. |
