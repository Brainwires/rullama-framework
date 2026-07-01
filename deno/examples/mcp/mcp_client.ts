// Example: MCP Client
// Demonstrates MCP client configuration, McpServerConfig construction,
// McpConfigManager usage, and the full McpClient API surface.
// Since we cannot connect to a real MCP server in this example, we show
// configuration patterns and API overview using mock data.
// Run: deno run --allow-read --allow-write --allow-env deno/examples/mcp/mcp_client.ts

import {
  McpClient,
  McpConfigManager,
  type McpServerConfig,
} from "@rullama/mcp-client";

async function main(): Promise<void> {
  console.log("=== MCP Client Example ===\n");

  // -----------------------------------------------------------------------
  // 1. Configure MCP servers
  // -----------------------------------------------------------------------
  console.log("--- 1. Server Configuration ---");

  const filesystemServer: McpServerConfig = {
    name: "filesystem",
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
  };
  console.log(`  Server: ${filesystemServer.name}`);
  console.log(
    `  Command: ${filesystemServer.command} ${filesystemServer.args.join(" ")}`,
  );
  console.log();

  // Server with environment variables
  const githubServer: McpServerConfig = {
    name: "github",
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-github"],
    env: { GITHUB_TOKEN: "ghp_demo_token" },
  };
  console.log(`  Server: ${githubServer.name}`);
  console.log(
    `  Command: ${githubServer.command} ${githubServer.args.join(" ")}`,
  );
  console.log(
    `  Env vars: [${Object.keys(githubServer.env ?? {}).join(", ")}]`,
  );
  console.log();

  // Configs serialize to JSON for persistence
  const json = JSON.stringify(filesystemServer, null, 2);
  console.log("  Serialized config:");
  console.log(`  ${json}`);
  console.log();

  // -----------------------------------------------------------------------
  // 2. Create the MCP client
  // -----------------------------------------------------------------------
  console.log("--- 2. Create McpClient ---");

  const client = new McpClient("rullama-example", "0.1.0");
  console.log(
    `  Created McpClient (name="${client.clientName}", version="${client.clientVersion}")`,
  );
  console.log(`  Connected servers: [${client.listConnected().join(", ")}]`);
  console.log();

  // Also show the default factory
  const defaultClient = McpClient.createDefault();
  console.log(
    `  Default McpClient: name="${defaultClient.clientName}", version="${defaultClient.clientVersion}"`,
  );
  console.log();

  // -----------------------------------------------------------------------
  // 3. Connection API overview (without real servers)
  // -----------------------------------------------------------------------
  console.log("--- 3. Connection API Overview ---");
  console.log("  The McpClient provides these methods:\n");

  console.log(
    "  // Connect to a server (spawns process, performs MCP handshake)",
  );
  console.log("  await client.connect(serverConfig);\n");

  console.log("  // Check connection status");
  console.log('  client.isConnected("filesystem");\n');

  console.log("  // List all connected servers");
  console.log("  client.listConnected();\n");
  console.log();

  // -----------------------------------------------------------------------
  // 4. Tool API
  // -----------------------------------------------------------------------
  console.log("--- 4. Tool API ---");
  console.log("  // List available tools from a connected server");
  console.log('  const tools = await client.listTools("filesystem");');
  console.log("  for (const tool of tools) {");
  console.log("    console.log(`Tool: ${tool.name} - ${tool.description}`);");
  console.log("  }\n");

  console.log("  // Call a tool with JSON arguments");
  console.log("  const result = await client.callTool(");
  console.log('    "filesystem",');
  console.log('    "read_file",');
  console.log('    { path: "/tmp/test.txt" },');
  console.log("  );\n");

  console.log("  // Call a tool with notification forwarding");
  console.log("  const result2 = await client.callToolWithNotifications(");
  console.log('    "filesystem",');
  console.log('    "read_file",');
  console.log('    { path: "/tmp/test.txt" },');
  console.log("    (notification) => console.log(notification),");
  console.log("  );\n");

  // Show mock tool arguments
  const mockArgs = {
    path: "/tmp/example.txt",
    encoding: "utf-8",
  };
  console.log("  Mock tool arguments:");
  console.log(`  ${JSON.stringify(mockArgs, null, 2)}`);
  console.log();

  // -----------------------------------------------------------------------
  // 5. Resources and prompts API
  // -----------------------------------------------------------------------
  console.log("--- 5. Resources & Prompts API ---");
  console.log("  // List resources exposed by a server");
  console.log(
    '  const resources = await client.listResources("filesystem");\n',
  );

  console.log("  // Read a specific resource by URI");
  console.log(
    '  const content = await client.readResource("filesystem", "file:///tmp/data.json");\n',
  );

  console.log("  // List prompt templates");
  console.log('  const prompts = await client.listPrompts("github");\n');

  console.log("  // Get a prompt with arguments");
  console.log("  const prompt = await client.getPrompt(");
  console.log('    "github",');
  console.log('    "review_pr",');
  console.log("    { pr_number: 42 },");
  console.log("  );\n");
  console.log();

  // -----------------------------------------------------------------------
  // 6. Server info and capabilities
  // -----------------------------------------------------------------------
  console.log("--- 6. Server Info & Capabilities ---");
  console.log("  // After connecting, query server metadata");
  console.log('  const info = client.getServerInfo("filesystem");');
  console.log("  // => { name: string, version: string }\n");

  console.log('  const caps = client.getCapabilities("filesystem");');
  console.log("  // Capabilities indicate which features the server supports:");
  console.log("  // - tools: server exposes callable tools");
  console.log("  // - resources: server exposes readable resources");
  console.log("  // - prompts: server exposes prompt templates\n");
  console.log();

  // -----------------------------------------------------------------------
  // 7. Disconnection and cancellation
  // -----------------------------------------------------------------------
  console.log("--- 7. Cleanup ---");
  console.log("  // Disconnect from a specific server");
  console.log('  await client.disconnect("filesystem");\n');

  console.log("  // Cancel a pending request (JSON-RPC $/cancelRequest)");
  console.log('  await client.cancelRequest("github", requestId);\n');

  // Verify no connections remain
  console.log(
    `  Final connected servers: [${client.listConnected().join(", ")}]`,
  );

  console.log(
    "\nDone! To connect to real MCP servers, ensure the server",
  );
  console.log(
    "command is installed (e.g., `npm install -g @modelcontextprotocol/server-filesystem`)",
  );
  console.log(
    "and call client.connect(config) to establish a connection.",
  );
}

await main();
