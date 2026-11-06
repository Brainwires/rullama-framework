/**
 * Tests for McpToolRegistry.
 * Equivalent to Rust registry::tests.
 */

import { assert, assertEquals, assertFalse, assertRejects } from "@std/assert";
import type { McpHandler } from "./handler.ts";
import { McpToolRegistry } from "./registry.ts";
import { McpServer, RequestContext } from "./server.ts";
import { MemoryTransport } from "./test_util.ts";

// deno-lint-ignore require-await
const echoHandler = async (
  _args: Record<string, unknown>,
  _ctx: RequestContext,
) => ({
  content: [],
  isError: false,
});

Deno.test("registry - register and list", () => {
  const registry = new McpToolRegistry();
  registry.register("echo", "Echo tool", { type: "object" }, echoHandler);

  const tools = registry.listTools();
  assertEquals(tools.length, 1);
  assertEquals(tools[0].name, "echo");
});

Deno.test("registry - hasTool", () => {
  const registry = new McpToolRegistry();
  registry.register("test", "Test tool", { type: "object" }, echoHandler);

  assert(registry.hasTool("test"));
  assertFalse(registry.hasTool("nonexistent"));
});

Deno.test("registry - dispatch calls handler", async () => {
  const registry = new McpToolRegistry();
  registry.register(
    "greet",
    "Greet tool",
    { type: "object" },
    // deno-lint-ignore require-await
    async (args: Record<string, unknown>) => ({
      content: [{ type: "text" as const, text: `Hello ${args.name}` }],
      isError: false,
    }),
  );

  const ctx = new RequestContext(1);
  const result = await registry.dispatch("greet", { name: "world" }, ctx);
  assertEquals(result.content.length, 1);
});

Deno.test("registry - dispatch unknown tool throws", async () => {
  const registry = new McpToolRegistry();
  const ctx = new RequestContext(1);

  await assertRejects(
    () => registry.dispatch("nonexistent", {}, ctx),
    Error,
    "Tool not found",
  );
});

Deno.test("registry - register without options lists only the required fields", () => {
  const registry = new McpToolRegistry();
  registry.register("echo", "Echo tool", { type: "object" }, echoHandler);
  assertEquals(registry.listTools(), [
    { name: "echo", description: "Echo tool", inputSchema: { type: "object" } },
  ]);
});

Deno.test("registry - register sets annotations and the other optional Tool fields", () => {
  const registry = new McpToolRegistry();
  registry.register("rm", "Delete a file", { type: "object" }, echoHandler, {
    title: "Delete file",
    annotations: {
      title: "Delete",
      readOnlyHint: false,
      destructiveHint: true,
      idempotentHint: true,
      openWorldHint: false,
    },
    outputSchema: { type: "object" },
    icons: [{ src: "data:image/png;base64,AA==" }],
    _meta: { "x/y": 1 },
  });
  assertEquals(registry.listTools(), [{
    name: "rm",
    description: "Delete a file",
    inputSchema: { type: "object" },
    title: "Delete file",
    annotations: {
      title: "Delete",
      readOnlyHint: false,
      destructiveHint: true,
      idempotentHint: true,
      openWorldHint: false,
    },
    outputSchema: { type: "object" },
    icons: [{ src: "data:image/png;base64,AA==" }],
    _meta: { "x/y": 1 },
  }]);
});

Deno.test("registry - undefined and unknown option keys are dropped", () => {
  const registry = new McpToolRegistry();
  registry.register("t", "T", {}, echoHandler, {
    title: undefined,
    annotations: { readOnlyHint: true },
    ...({ name: "spoofed", inputSchema: { evil: true } } as object),
  });
  assertEquals(registry.listTools(), [
    {
      name: "t",
      description: "T",
      inputSchema: {},
      annotations: { readOnlyHint: true },
    },
  ]);
});

Deno.test("registry - tools/list emits registered annotations", async () => {
  const registry = new McpToolRegistry();
  registry.register("read", "Read", { type: "object" }, echoHandler, {
    annotations: { readOnlyHint: true, openWorldHint: false },
  });
  registry.register("plain", "Plain", { type: "object" }, echoHandler);
  const handler: McpHandler = {
    serverInfo: () => ({ name: "reg", version: "0" }),
    capabilities: () => ({ tools: {} }),
    listTools: () => registry.listTools(),
    callTool: (name, args, ctx) => registry.dispatch(name, args, ctx),
  };
  const t = new MemoryTransport();
  const session = new McpServer(handler).serveConnection(t);
  t.push({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "c", version: "1" },
    },
  });
  t.push({ jsonrpc: "2.0", id: 2, method: "tools/list" });
  const [read, plain] = (await t.response(2)).result.tools;
  assertEquals(read.annotations, { readOnlyHint: true, openWorldHint: false });
  assertFalse("annotations" in plain);
  t.end();
  await session.done;
});
