// Example: MCP Server — tool registry, middleware pipeline, and server construction
// Demonstrates building an McpToolRegistry with tool handlers, assembling a
// MiddlewareChain (auth, logging, rate-limiting, tool filtering), dispatching
// tool calls, and constructing an McpServer with middleware layers.
// Run: deno run deno/examples/agent-network/mcp_server.ts

import {
  AuthMiddleware,
  LoggingMiddleware,
  type McpHandler,
  McpServer,
  type McpToolDef,
  McpToolRegistry,
  MiddlewareChain,
  RateLimitMiddleware,
  RequestContext,
  ToolFilterMiddleware,
  type ToolHandler,
} from "@rullama/network";

import type {
  CallToolResult,
  ServerCapabilities,
  ServerInfo,
} from "@rullama/mcp-client";

async function main(): Promise<void> {
  console.log("=== MCP Server Example ===\n");

  // ---------------------------------------------------------------------------
  // 1. Build the tool registry with handlers
  // ---------------------------------------------------------------------------
  console.log("--- Tool Registry ---");

  const registry = new McpToolRegistry();

  // Echo handler: returns the input arguments as text
  const echoHandler: ToolHandler = async (
    args: Record<string, unknown>,
    _ctx: RequestContext,
  ): Promise<CallToolResult> => {
    const message = (args.message as string) ?? "(empty)";
    console.log(`    [EchoHandler] called with: ${message}`);
    return {
      content: [{ type: "text", text: `Echo: ${message}` }],
    };
  };

  registry.register(
    "echo",
    "Echoes back the provided arguments",
    {
      type: "object",
      properties: {
        message: { type: "string", description: "The message to echo" },
      },
      required: ["message"],
    },
    echoHandler,
  );

  // Time handler: returns the current UTC time
  const timeHandler: ToolHandler = async (
    _args: Record<string, unknown>,
    _ctx: RequestContext,
  ): Promise<CallToolResult> => {
    const now = new Date().toISOString();
    console.log(`    [TimeHandler] returning time: ${now}`);
    return {
      content: [{ type: "text", text: now }],
    };
  };

  registry.register(
    "get_time",
    "Returns the current UTC time",
    { type: "object", properties: {} },
    timeHandler,
  );

  for (const def of registry.listTools()) {
    console.log(`  Registered tool: ${def.name} — ${def.description}`);
  }
  console.log(`  Has "echo": ${registry.hasTool("echo")}`);
  console.log(`  Has "missing": ${registry.hasTool("missing")}`);
  console.log();

  // ---------------------------------------------------------------------------
  // 2. Build the middleware chain
  // ---------------------------------------------------------------------------
  console.log("--- Middleware Chain ---");

  const chain = new MiddlewareChain();

  // Auth: require a bearer token for non-initialize requests
  chain.add(new AuthMiddleware("demo-secret-token"));
  console.log("  1. AuthMiddleware       (token = demo-secret-token)");

  // Logging: log every request
  chain.add(new LoggingMiddleware());
  console.log("  2. LoggingMiddleware");

  // Rate limit: 20 requests per second
  chain.add(new RateLimitMiddleware(20));
  console.log("  3. RateLimitMiddleware  (20 req/s)");

  // Tool filter: block dangerous tools
  chain.add(ToolFilterMiddleware.deny(["rm_rf", "drop_database"]));
  console.log("  4. ToolFilterMiddleware (deny: rm_rf, drop_database)");
  console.log();

  // ---------------------------------------------------------------------------
  // 3. Dispatch tool calls through the registry
  // ---------------------------------------------------------------------------
  console.log("--- Dispatch Test ---");

  const ctx = new RequestContext(1);

  console.log("  Dispatching 'echo':");
  try {
    const result = await registry.dispatch("echo", { message: "hello" }, ctx);
    console.log(
      `    Result: OK — ${
        result.content[0]?.type === "text" ? result.content[0].text : "?"
      }`,
    );
  } catch (e) {
    console.log(`    Result: Error — ${e}`);
  }

  console.log("  Dispatching 'get_time':");
  try {
    const result = await registry.dispatch("get_time", {}, ctx);
    console.log(
      `    Result: OK — ${
        result.content[0]?.type === "text" ? result.content[0].text : "?"
      }`,
    );
  } catch (e) {
    console.log(`    Result: Error — ${e}`);
  }

  console.log("  Dispatching 'nonexistent_tool':");
  try {
    await registry.dispatch("nonexistent_tool", {}, ctx);
    console.log("    Result: OK (unexpected)");
  } catch (e) {
    console.log(`    Result: Error — ${e}`);
  }
  console.log();

  // ---------------------------------------------------------------------------
  // 4. Process requests through the middleware chain
  // ---------------------------------------------------------------------------
  console.log("--- Middleware Processing ---");

  // Simulate an initialize request (should pass auth)
  const initRequest = {
    jsonrpc: "2.0" as const,
    id: 1,
    method: "initialize",
    params: undefined,
  };
  const initCtx = new RequestContext(1);
  const initResult = await chain.processRequest(initRequest, initCtx);
  console.log(
    `  initialize -> ${
      initResult ? `Rejected: ${initResult.message}` : "Continue"
    }`,
  );

  // Simulate a tools/call request (will be rejected by auth — no token)
  const toolRequest = {
    jsonrpc: "2.0" as const,
    id: 2,
    method: "tools/call",
    params: { name: "echo", arguments: { message: "test" } },
  };
  const toolCtx = new RequestContext(2);
  const toolResult = await chain.processRequest(toolRequest, toolCtx);
  console.log(
    `  tools/call 'echo' (no token) -> ${
      toolResult ? `Rejected: ${toolResult.message}` : "Continue"
    }`,
  );

  // Simulate a tools/call with valid auth token
  const authToolRequest = {
    jsonrpc: "2.0" as const,
    id: 3,
    method: "tools/call",
    params: {
      name: "echo",
      arguments: { message: "test" },
      _auth_token: "demo-secret-token",
    },
  };
  const authToolCtx = new RequestContext(3);
  const authToolResult = await chain.processRequest(
    authToolRequest,
    authToolCtx,
  );
  console.log(
    `  tools/call 'echo' (with token) -> ${
      authToolResult ? `Rejected: ${authToolResult.message}` : "Continue"
    }`,
  );
  console.log();

  // ---------------------------------------------------------------------------
  // 5. Construct an McpServer (without running the event loop)
  // ---------------------------------------------------------------------------
  console.log("--- Server Construction ---");

  // Build a handler that delegates to the registry
  const handler: McpHandler = {
    serverInfo(): ServerInfo {
      return { name: "demo-mcp-server", version: "0.1.0" };
    },
    capabilities(): ServerCapabilities {
      return {};
    },
    listTools(): McpToolDef[] {
      return registry.listTools();
    },
    async callTool(
      name: string,
      args: Record<string, unknown>,
      reqCtx: RequestContext,
    ): Promise<CallToolResult> {
      return registry.dispatch(name, args, reqCtx);
    },
  };

  const server = new McpServer(handler)
    .withMiddleware(new AuthMiddleware("secret"))
    .withMiddleware(new LoggingMiddleware())
    .withMiddleware(new RateLimitMiddleware(10))
    .withMiddleware(ToolFilterMiddleware.deny(["rm_rf"]));

  console.log("  McpServer created with handler + 4 middleware layers");
  console.log("  Server info:", JSON.stringify(handler.serverInfo()));
  console.log("  (Skipping server.run() — requires a connected transport)");

  // Prevent unused-variable lint
  void server;

  console.log("\nDone.");
}

await main();
