/**
 * Tests for ToolFilterMiddleware (equivalent to Rust tool_filter::tests) and
 * IdentityToolFilterMiddleware.
 */

import { assert, assertEquals, assertFalse } from "@std/assert";
import type { CallToolResult } from "@rullama/mcp-client";
import {
  IdentityToolFilterMiddleware,
  ToolFilterMiddleware,
} from "./tool_filter.ts";
import { McpServer, RequestContext } from "../server.ts";
import type { McpHandler } from "../handler.ts";
import { MemoryTransport } from "../test_util.ts";

Deno.test("tool filter - allow list", () => {
  const filter = ToolFilterMiddleware.allowOnly(["agent_spawn", "agent_list"]);
  assert(filter.isAllowed("agent_spawn"));
  assert(filter.isAllowed("agent_list"));
  assertFalse(filter.isAllowed("bash"));
});

Deno.test("tool filter - deny list", () => {
  const filter = ToolFilterMiddleware.deny(["bash", "write_file"]);
  assertFalse(filter.isAllowed("bash"));
  assertFalse(filter.isAllowed("write_file"));
  assert(filter.isAllowed("agent_spawn"));
});

// --- IdentityToolFilterMiddleware ------------------------------------------

const tools = ["read", "write", "admin_reset"];

function toolHandler(): McpHandler {
  return {
    serverInfo: () => ({ name: "f", version: "0" }),
    capabilities: () => ({ tools: {} }),
    listTools: () =>
      tools.map((name) => ({ name, description: "", inputSchema: {} })),
    callTool: (name): Promise<CallToolResult> =>
      Promise.resolve({ content: [{ type: "text", text: `ran ${name}` }] }),
  };
}

/** List + call every tool on a connection with `identity`. */
async function probe(mw: IdentityToolFilterMiddleware, identity?: unknown) {
  const t = new MemoryTransport();
  const session = new McpServer(toolHandler()).withMiddleware(mw)
    .serveConnection(t, {
      connection: { id: "c", transport: "test", identity },
    });
  t.push({ jsonrpc: "2.0", id: "l", method: "tools/list" });
  const listed = (await t.response("l")).result.tools.map((
    x: { name: string },
  ) => x.name);
  const called: Record<string, string> = {};
  for (const [i, name] of tools.entries()) {
    t.push({
      jsonrpc: "2.0",
      id: i,
      method: "tools/call",
      params: { name, arguments: {} },
    });
    const r = await t.response(i);
    called[name] = r.error ? `error ${r.error.code}` : r.result.content[0].text;
  }
  t.end();
  await session.done;
  return { listed, called };
}

Deno.test("identity filter hides and rejects tools per connection identity", async () => {
  const mw = new IdentityToolFilterMiddleware((tool, identity) =>
    (identity as { role?: string } | undefined)?.role === "admin" ||
    !tool.startsWith("admin_")
  );
  const admin = await probe(mw, { role: "admin" });
  assertEquals(admin.listed, tools);
  assertEquals(admin.called.admin_reset, "ran admin_reset");

  const user = await probe(mw, { role: "user" });
  assertEquals(user.listed, ["read", "write"]);
  assertEquals(user.called.read, "ran read");
  assertEquals(user.called.admin_reset, "error -32001");

  const anonymous = await probe(mw);
  assertEquals(anonymous.listed, ["read", "write"]);
});

Deno.test("identity filter allowFor maps identities to tool lists", async () => {
  const mw = IdentityToolFilterMiddleware.allowFor((identity) => {
    const tile = (identity as { tile?: string } | undefined)?.tile;
    if (tile === "root") return "*";
    if (tile === "ro") return ["read"];
    return undefined;
  });
  assertEquals((await probe(mw, { tile: "root" })).listed, tools);
  const ro = await probe(mw, { tile: "ro" });
  assertEquals(ro.listed, ["read"]);
  assertEquals(ro.called.write, "error -32001");
  assertEquals((await probe(mw, { tile: "other" })).listed, []);
});

Deno.test("identity filter fails closed when the predicate throws, and supports async", async () => {
  const broken = new IdentityToolFilterMiddleware(() => {
    throw new Error("policy store down");
  });
  const r = await probe(broken, "x");
  assertEquals(r.listed, []);
  assertEquals(r.called.read, "error -32001");

  const asyncMw = new IdentityToolFilterMiddleware(async (tool) => {
    await Promise.resolve();
    return tool === "write";
  });
  assertEquals((await probe(asyncMw)).listed, ["write"]);
  const ctx = new RequestContext(1);
  assert(await asyncMw.isAllowed("write", ctx));
  assertFalse(await asyncMw.isAllowed("read", ctx));
});
