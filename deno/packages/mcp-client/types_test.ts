/**
 * Tests for JSON-RPC and MCP protocol types.
 * Mirrors Rust tests in `rullama-mcp/src/types.rs`.
 */

import { assertEquals, assertExists } from "@std/assert";
import {
  createJsonRpcNotification,
  createJsonRpcRequest,
  isJsonRpcNotification,
  isJsonRpcResponse,
  parseJsonRpcMessage,
  parseNotification,
} from "./types.ts";
import type {
  CallToolParams,
  CallToolResult,
  JsonRpcError,
  JsonRpcResponse,
  McpPrompt,
  McpResource,
  McpTool,
  ProgressParams,
} from "./types.ts";

// =============================================================================
// JSON-RPC Request tests — mirrors Rust test_json_rpc_request_new
// =============================================================================

Deno.test("JsonRpcRequest - create with params", () => {
  const request = createJsonRpcRequest(1, "test_method", { key: "value" });

  assertEquals(request.jsonrpc, "2.0");
  assertEquals(request.id, 1);
  assertEquals(request.method, "test_method");
  assertExists(request.params);
});

Deno.test("JsonRpcRequest - create without params", () => {
  const request = createJsonRpcRequest(1, "test");

  assertEquals(request.jsonrpc, "2.0");
  assertEquals(request.id, 1);
  assertEquals(request.method, "test");
  assertEquals(request.params, undefined);
});

// Mirrors Rust test_json_rpc_request_serialization
Deno.test("JsonRpcRequest - serialization", () => {
  const request = createJsonRpcRequest(1, "test");
  const json = JSON.stringify(request);

  assertEquals(json.includes("jsonrpc"), true);
  assertEquals(json.includes("2.0"), true);
  assertEquals(json.includes("test"), true);
});

Deno.test("JsonRpcRequest - serialization excludes undefined params", () => {
  const request = createJsonRpcRequest(1, "test");
  const json = JSON.stringify(request);
  const parsed = JSON.parse(json);

  assertEquals("params" in parsed, false);
});

Deno.test("JsonRpcRequest - serialization includes params when present", () => {
  const request = createJsonRpcRequest(1, "test", { foo: "bar" });
  const json = JSON.stringify(request);
  const parsed = JSON.parse(json);

  assertEquals(parsed.params.foo, "bar");
});

// =============================================================================
// JSON-RPC Response tests — mirrors Rust test_json_rpc_response_success/error
// =============================================================================

Deno.test("JsonRpcResponse - success", () => {
  const response: JsonRpcResponse = {
    jsonrpc: "2.0",
    id: 1,
    result: { status: "ok" },
  };

  assertExists(response.result);
  assertEquals(response.error, undefined);
});

Deno.test("JsonRpcResponse - error", () => {
  const error: JsonRpcError = {
    code: -32600,
    message: "Invalid Request",
  };
  const response: JsonRpcResponse = {
    jsonrpc: "2.0",
    id: 1,
    error,
  };

  assertEquals(response.result, undefined);
  assertExists(response.error);
  assertEquals(response.error!.code, -32600);
  assertEquals(response.error!.message, "Invalid Request");
});

Deno.test("JsonRpcError - with data", () => {
  const error: JsonRpcError = {
    code: -32603,
    message: "Internal error",
    data: { details: "something went wrong" },
  };

  assertEquals(error.code, -32603);
  assertExists(error.data);
});

// =============================================================================
// JSON-RPC Notification tests
// =============================================================================

Deno.test("JsonRpcNotification - create with params", () => {
  const notif = createJsonRpcNotification("test/notify", { key: "value" });

  assertEquals(notif.jsonrpc, "2.0");
  assertEquals(notif.method, "test/notify");
  assertExists(notif.params);
});

Deno.test("JsonRpcNotification - create without params", () => {
  const notif = createJsonRpcNotification("test/notify");

  assertEquals(notif.jsonrpc, "2.0");
  assertEquals(notif.method, "test/notify");
  assertEquals(notif.params, undefined);
});

// =============================================================================
// JsonRpcMessage parsing tests
// =============================================================================

Deno.test("parseJsonRpcMessage - response with id", () => {
  const raw = JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    result: { status: "ok" },
  });
  const msg = parseJsonRpcMessage(raw);

  assertEquals(isJsonRpcResponse(msg), true);
  assertEquals(isJsonRpcNotification(msg), false);
  if (msg.type === "response") {
    assertEquals(msg.response.id, 1);
  }
});

Deno.test("parseJsonRpcMessage - notification without id", () => {
  const raw = JSON.stringify({
    jsonrpc: "2.0",
    method: "notifications/progress",
    params: { progressToken: "abc", progress: 50 },
  });
  const msg = parseJsonRpcMessage(raw);

  assertEquals(isJsonRpcNotification(msg), true);
  assertEquals(isJsonRpcResponse(msg), false);
  if (msg.type === "notification") {
    assertEquals(msg.notification.method, "notifications/progress");
  }
});

Deno.test("parseJsonRpcMessage - null id treated as notification", () => {
  const raw = JSON.stringify({
    jsonrpc: "2.0",
    id: null,
    method: "notifications/initialized",
  });
  const msg = parseJsonRpcMessage(raw);

  assertEquals(isJsonRpcNotification(msg), true);
});

// =============================================================================
// MCP Notification parsing tests
// =============================================================================

Deno.test("parseNotification - progress notification", () => {
  const notif = createJsonRpcNotification("notifications/progress", {
    progressToken: "token-1",
    progress: 50,
    total: 100,
    message: "Halfway done",
  });

  const parsed = parseNotification(notif);
  assertEquals(parsed.type, "progress");
  if (parsed.type === "progress") {
    assertEquals(parsed.params.progressToken, "token-1");
    assertEquals(parsed.params.progress, 50);
    assertEquals(parsed.params.total, 100);
    assertEquals(parsed.params.message, "Halfway done");
  }
});

Deno.test("parseNotification - unknown notification", () => {
  const notif = createJsonRpcNotification("custom/event", { foo: "bar" });

  const parsed = parseNotification(notif);
  assertEquals(parsed.type, "unknown");
  if (parsed.type === "unknown") {
    assertEquals(parsed.method, "custom/event");
  }
});

// =============================================================================
// MCP Type shape tests — mirrors Rust test_type_aliases_work
// =============================================================================

Deno.test("McpTool - type shape", () => {
  const tool: McpTool = {
    name: "test-tool",
    description: "A test tool",
    inputSchema: {
      type: "object",
      properties: { input: { type: "string" } },
    },
  };

  assertEquals(tool.name, "test-tool");
  assertExists(tool.inputSchema);
});

Deno.test("McpResource - type shape", () => {
  const resource: McpResource = {
    uri: "file:///test.txt",
    name: "test.txt",
    description: "A test resource",
    mimeType: "text/plain",
  };

  assertEquals(resource.uri, "file:///test.txt");
  assertEquals(resource.name, "test.txt");
});

Deno.test("McpPrompt - type shape", () => {
  const prompt: McpPrompt = {
    name: "test-prompt",
    description: "A test prompt",
    arguments: [
      { name: "topic", description: "The topic", required: true },
    ],
  };

  assertEquals(prompt.name, "test-prompt");
  assertEquals(prompt.arguments?.length, 1);
});

Deno.test("CallToolParams - type shape", () => {
  const params: CallToolParams = {
    name: "echo",
    arguments: { message: "hello" },
  };

  assertEquals(params.name, "echo");
  assertEquals(params.arguments?.message, "hello");
});

Deno.test("CallToolResult - text content", () => {
  const result: CallToolResult = {
    content: [{ type: "text", text: "Hello, world!" }],
    isError: false,
  };

  assertEquals(result.content.length, 1);
  assertEquals(result.content[0].type, "text");
  if (result.content[0].type === "text") {
    assertEquals(result.content[0].text, "Hello, world!");
  }
});

Deno.test("ProgressParams - type shape", () => {
  const params: ProgressParams = {
    progressToken: "abc",
    progress: 42,
    total: 100,
    message: "Processing...",
  };

  assertEquals(params.progressToken, "abc");
  assertEquals(params.progress, 42);
});

// =============================================================================
// Roundtrip serialization tests
// =============================================================================

Deno.test("JsonRpcRequest - roundtrip serialization", () => {
  const original = createJsonRpcRequest(42, "tools/call", {
    name: "echo",
    arguments: { msg: "hello" },
  });
  const json = JSON.stringify(original);
  const parsed = JSON.parse(json);

  assertEquals(parsed.jsonrpc, "2.0");
  assertEquals(parsed.id, 42);
  assertEquals(parsed.method, "tools/call");
  assertEquals(parsed.params.name, "echo");
  assertEquals(parsed.params.arguments.msg, "hello");
});

Deno.test("JsonRpcResponse - roundtrip serialization", () => {
  const original: JsonRpcResponse = {
    jsonrpc: "2.0",
    id: 1,
    result: { tools: [{ name: "echo", inputSchema: { type: "object" } }] },
  };
  const json = JSON.stringify(original);
  const parsed = JSON.parse(json);

  assertEquals(parsed.jsonrpc, "2.0");
  assertEquals(parsed.id, 1);
  assertEquals(parsed.result.tools[0].name, "echo");
});
