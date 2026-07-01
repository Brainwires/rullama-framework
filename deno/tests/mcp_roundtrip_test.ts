/**
 * Cross-package integration test: MCP type serialization roundtrip.
 *
 * Verifies that @rullama/mcp types (McpTool, McpResource, McpPrompt)
 * can be serialized to JSON and deserialized with all fields preserved.
 * Also tests JSON-RPC request/response creation and parsing.
 */

import {
  assert,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  createJsonRpcNotification,
  createJsonRpcRequest,
  isJsonRpcNotification,
  isJsonRpcResponse,
  type JsonRpcRequest,
  type JsonRpcResponse,
  type McpPrompt,
  type McpResource,
  type McpTool,
  parseJsonRpcMessage,
  parseNotification,
  type PromptArgument,
} from "@rullama/mcp-client";

// ---------------------------------------------------------------------------
// McpTool roundtrip
// ---------------------------------------------------------------------------

Deno.test("McpTool serializes and deserializes correctly", () => {
  const tool: McpTool = {
    name: "code_search",
    description: "Search codebase using semantic queries",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        limit: { type: "number", description: "Max results" },
      },
      required: ["query"],
    },
  };

  const json = JSON.stringify(tool);
  const parsed: McpTool = JSON.parse(json);

  assertEquals(parsed.name, "code_search");
  assertEquals(parsed.description, "Search codebase using semantic queries");
  assertEquals(parsed.inputSchema.type, "object");
  assertEquals(
    (parsed.inputSchema.properties as Record<string, unknown>)["query"],
    { type: "string", description: "Search query" },
  );
  assertEquals(parsed.inputSchema.required, ["query"]);
});

// ---------------------------------------------------------------------------
// McpResource roundtrip
// ---------------------------------------------------------------------------

Deno.test("McpResource serializes and deserializes correctly", () => {
  const resource: McpResource = {
    uri: "file:///home/user/project/README.md",
    name: "README.md",
    description: "Project readme file",
    mimeType: "text/markdown",
  };

  const json = JSON.stringify(resource);
  const parsed: McpResource = JSON.parse(json);

  assertEquals(parsed.uri, "file:///home/user/project/README.md");
  assertEquals(parsed.name, "README.md");
  assertEquals(parsed.description, "Project readme file");
  assertEquals(parsed.mimeType, "text/markdown");
});

Deno.test("McpResource without optional fields roundtrips", () => {
  const resource: McpResource = {
    uri: "mem://buffer/scratch",
    name: "scratch",
  };

  const json = JSON.stringify(resource);
  const parsed: McpResource = JSON.parse(json);

  assertEquals(parsed.uri, "mem://buffer/scratch");
  assertEquals(parsed.name, "scratch");
  assertEquals(parsed.description, undefined);
  assertEquals(parsed.mimeType, undefined);
});

// ---------------------------------------------------------------------------
// McpPrompt roundtrip
// ---------------------------------------------------------------------------

Deno.test("McpPrompt with arguments serializes correctly", () => {
  const args: PromptArgument[] = [
    { name: "language", description: "Programming language", required: true },
    { name: "style", description: "Code style preference", required: false },
  ];

  const prompt: McpPrompt = {
    name: "code_review",
    description: "Review code and provide feedback",
    arguments: args,
  };

  const json = JSON.stringify(prompt);
  const parsed: McpPrompt = JSON.parse(json);

  assertEquals(parsed.name, "code_review");
  assertEquals(parsed.description, "Review code and provide feedback");
  assertEquals(parsed.arguments?.length, 2);
  assertEquals(parsed.arguments![0].name, "language");
  assertEquals(parsed.arguments![0].required, true);
  assertEquals(parsed.arguments![1].name, "style");
  assertEquals(parsed.arguments![1].required, false);
});

// ---------------------------------------------------------------------------
// JSON-RPC request creation
// ---------------------------------------------------------------------------

Deno.test("createJsonRpcRequest produces valid request", () => {
  const req = createJsonRpcRequest(1, "tools/list", { cursor: null });

  assertEquals(req.jsonrpc, "2.0");
  assertEquals(req.id, 1);
  assertEquals(req.method, "tools/list");
  assertEquals(req.params, { cursor: null });
});

Deno.test("createJsonRpcRequest without params omits field", () => {
  const req = createJsonRpcRequest(42, "initialize");

  assertEquals(req.jsonrpc, "2.0");
  assertEquals(req.id, 42);
  assertEquals(req.method, "initialize");
  assertEquals(req.params, undefined);
});

// ---------------------------------------------------------------------------
// JSON-RPC notification creation
// ---------------------------------------------------------------------------

Deno.test("createJsonRpcNotification produces valid notification", () => {
  const notif = createJsonRpcNotification("notifications/initialized");

  assertEquals(notif.jsonrpc, "2.0");
  assertEquals(notif.method, "notifications/initialized");
  assertEquals(notif.params, undefined);
});

// ---------------------------------------------------------------------------
// parseJsonRpcMessage discriminates response vs notification
// ---------------------------------------------------------------------------

Deno.test("parseJsonRpcMessage recognizes response", () => {
  const raw = JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    result: { tools: [] },
  });

  const msg = parseJsonRpcMessage(raw);
  assert(isJsonRpcResponse(msg));
  assertEquals(msg.response.id, 1);
  assertEquals(msg.response.result, { tools: [] });
});

Deno.test("parseJsonRpcMessage recognizes notification (no id)", () => {
  const raw = JSON.stringify({
    jsonrpc: "2.0",
    method: "notifications/progress",
    params: { progressToken: "tok1", progress: 50, total: 100 },
  });

  const msg = parseJsonRpcMessage(raw);
  assert(isJsonRpcNotification(msg));
  assertEquals(msg.notification.method, "notifications/progress");
});

// ---------------------------------------------------------------------------
// parseNotification typed parsing
// ---------------------------------------------------------------------------

Deno.test("parseNotification identifies progress notification", () => {
  const notif = createJsonRpcNotification("notifications/progress", {
    progressToken: "abc",
    progress: 75,
    total: 100,
    message: "Processing...",
  });

  const parsed = parseNotification(notif);
  assertEquals(parsed.type, "progress");
  if (parsed.type === "progress") {
    assertEquals(parsed.params.progressToken, "abc");
    assertEquals(parsed.params.progress, 75);
    assertEquals(parsed.params.total, 100);
    assertEquals(parsed.params.message, "Processing...");
  }
});

Deno.test("parseNotification falls back to unknown for other methods", () => {
  const notif = createJsonRpcNotification("custom/event", { data: 123 });

  const parsed = parseNotification(notif);
  assertEquals(parsed.type, "unknown");
  if (parsed.type === "unknown") {
    assertEquals(parsed.method, "custom/event");
  }
});
