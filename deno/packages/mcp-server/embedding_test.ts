// Tests for the embedding hooks: server/discover handling, first-line hooks
// on serveConnection, 2025-11-25 tool execution errors and tool passthrough.

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { CallToolResult } from "@rullama/mcp-client";
import type { McpHandler } from "./handler.ts";
import { McpServer, type RequestContext } from "./server.ts";
import { AuthMiddleware } from "./middleware/auth.ts";
import { AgentNetworkError } from "./error.ts";
import { MemoryTransport, type Msg } from "./test_util.ts";

const text = (t: string): CallToolResult => ({
  content: [{ type: "text", text: t }],
});

function handler(): McpHandler {
  return {
    serverInfo: () => ({ name: "embed", version: "0", description: "test" }),
    capabilities: () => ({ tools: {} }),
    listTools: () => [
      {
        name: "whoami",
        description: "identity",
        inputSchema: { type: "object" },
        title: "Who am I",
        icons: [{ src: "data:image/png;base64,AA==", mimeType: "image/png" }],
        annotations: { readOnlyHint: true },
      },
      { name: "fail", description: "throws", inputSchema: {} },
    ],
    callTool: (name: string, args, ctx: RequestContext) => {
      if (name === "whoami") {
        return Promise.resolve(
          text(JSON.stringify(ctx.connection?.identity ?? null)),
        );
      }
      if (name === "fail") {
        if (args.kind === "notfound") {
          return Promise.reject(AgentNetworkError.toolNotFound("ghost"));
        }
        if (args.kind === "params") {
          return Promise.reject(
            AgentNetworkError.invalidParams("path must be absolute"),
          );
        }
        return Promise.reject(new Error("disk on fire"));
      }
      return Promise.reject(AgentNetworkError.toolNotFound(name));
    },
  };
}

const init = (id: number, protocolVersion = "2025-11-25"): Msg => ({
  jsonrpc: "2.0",
  id,
  method: "initialize",
  params: {
    protocolVersion,
    capabilities: {},
    clientInfo: { name: "c", version: "1" },
  },
});

const call = (id: number, name: string, args: Msg = {}): Msg => ({
  jsonrpc: "2.0",
  id,
  method: "tools/call",
  params: { name, arguments: args },
});

// ---------------------------------------------------------------------------
// server/discover (MCP 2026-07-28 probe)
// ---------------------------------------------------------------------------

Deno.test("server/discover gets MethodNotFound so a dual-era client falls back to initialize", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(handler()).serveConnection(t);
  // What Claude Code 2.1.x sends first.
  t.push({
    jsonrpc: "2.0",
    id: 0,
    method: "server/discover",
    params: {
      _meta: { "io.modelcontextprotocol/protocolVersion": "2026-07-28" },
    },
  });
  const r = await t.response(0);
  assertEquals(r.error.code, -32601);
  assert(r.error.message.includes("server/discover"));
  assert(r.error.message.includes("2025-11-25"), r.error.message);
  assert(!("result" in r));
  // Not a modern error code (UnsupportedProtocolVersion is -32022).
  assert(r.error.code !== -32022);

  t.push(init(1, "2026-07-28"));
  const ir = await t.response(1);
  assertEquals(ir.result.protocolVersion, "2025-11-25");
  assertEquals(ir.result.serverInfo.description, "test");
  t.end();
  await session.done;
});

Deno.test("server/discover is answered before middleware and never as a notification", async () => {
  const t = new MemoryTransport();
  const server = new McpServer(handler()).withMiddleware(
    new AuthMiddleware("secret"),
  );
  const session = server.serveConnection(t);
  t.push({ jsonrpc: "2.0", method: "server/discover" }); // notification
  t.push({ jsonrpc: "2.0", id: "d", method: "server/discover" });
  const r = await t.response("d");
  assertEquals(r.error.code, -32601); // not the auth middleware's -32003
  t.end();
  await session.done;
  assertEquals(t.written.length, 1);
});

// ---------------------------------------------------------------------------
// preAuth / firstLine on serveConnection
// ---------------------------------------------------------------------------

Deno.test("serveConnection preAuth admits with an identity and consumes the line", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(handler()).serveConnection(t, {
    preAuth: (line) =>
      line === "tok" ? { ok: true, identity: { tile: 7 } } : { ok: false },
  });
  t.push("tok");
  t.push(init(1));
  t.push(call(2, "whoami"));
  assertEquals((await t.response(2)).result.content[0].text, '{"tile":7}');
  assertEquals(session.connection.identity, { tile: 7 });
  assertEquals(t.written.length, 2); // nothing written for the token line
  t.end();
  await session.done;
});

Deno.test("serveConnection preAuth rejection writes Unauthorized and closes", async () => {
  const t = new MemoryTransport();
  let shutdowns = 0;
  const h = {
    ...handler(),
    onShutdown: () => (shutdowns++, Promise.resolve()),
  };
  const session = new McpServer(h).serveConnection(t, {
    preAuth: () => ({ ok: false, reason: "who are you" }),
  });
  t.push(init(1));
  await session.done;
  assertEquals(t.written.length, 1);
  assertEquals(t.written[0].error.code, -32003);
  assertEquals(t.written[0].error.message, "Unauthorized: who are you");
  assert(t.closed);
  assertEquals(shutdowns, 1);
});

Deno.test("serveConnection preAuth times out a silent client", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(handler()).serveConnection(t, {
    preAuth: (line) => ({ ok: line !== null }),
    firstLineTimeoutMs: 20,
  });
  await session.done;
  assertEquals(t.written[0].error.code, -32003);
  assert(t.closed);
});

Deno.test("firstLine can answer a host one-shot protocol and close", async () => {
  const t = new MemoryTransport();
  const seen: string[] = [];
  const session = new McpServer(handler()).serveConnection(t, {
    connection: { id: "tile-1", transport: "unix-ffi" },
    firstLine: (line, info) => {
      const msg = JSON.parse(line);
      if (msg.qutty === "hook") {
        seen.push(`${info.id}:${msg.event}`);
        return { action: "close", reply: JSON.stringify({ ok: true }) };
      }
      return { action: "mcp" };
    },
  });
  t.push({ qutty: "hook", event: "Stop" });
  await session.done;
  assertEquals(seen, ["tile-1:Stop"]);
  assertEquals(t.written, [{ ok: true }]);
  assert(t.closed);
});

Deno.test("firstLine hands a JSON-RPC first line back to MCP", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(handler()).serveConnection(t, {
    firstLine: (line) =>
      line.includes('"qutty"')
        ? { action: "close" }
        : { action: "mcp", identity: { via: "mcp" } },
  });
  t.push(init(1));
  assertEquals((await t.response(1)).result.protocolVersion, "2025-11-25");
  t.push(call(2, "whoami"));
  assertEquals((await t.response(2)).result.content[0].text, '{"via":"mcp"}');
  t.end();
  await session.done;
});

Deno.test("firstLine can consume a preamble or rewrite the line", async () => {
  const t1 = new MemoryTransport();
  const s1 = new McpServer(handler()).serveConnection(t1, {
    firstLine: () => ({ action: "consume", identity: "pre" }),
  });
  t1.push("HELLO tile=3");
  t1.push(init(1));
  assertEquals((await t1.response(1)).id, 1);
  t1.end();
  await s1.done;
  assertEquals(t1.written.length, 1);
  assertEquals(s1.connection.identity, "pre");

  const t2 = new MemoryTransport();
  const s2 = new McpServer(handler()).serveConnection(t2, {
    firstLine: () => ({ action: "mcp", line: JSON.stringify(init(9)) }),
  });
  t2.push("not json at all");
  assertEquals((await t2.response(9)).result.serverInfo.name, "embed");
  t2.end();
  await s2.done;
});

Deno.test("a throwing firstLine hook closes the connection; EOF skips the hook", async () => {
  const t = new MemoryTransport();
  const s = new McpServer(handler()).serveConnection(t, {
    firstLine: () => {
      throw new Error("bad hook");
    },
  });
  t.push(init(1));
  await s.done;
  assertEquals(t.written.length, 0);
  assert(t.closed);

  let called = false;
  const t2 = new MemoryTransport();
  const s2 = new McpServer(handler()).serveConnection(t2, {
    firstLine: () => {
      called = true;
      return { action: "mcp" };
    },
  });
  t2.end();
  await s2.done;
  assert(!called);
});

Deno.test("preAuth and firstLine together are rejected", () => {
  assertThrows(
    () =>
      new McpServer(handler()).serveConnection(new MemoryTransport(), {
        preAuth: () => ({ ok: true }),
        firstLine: () => ({ action: "mcp" }),
      }),
    TypeError,
    "mutually exclusive",
  );
});

// ---------------------------------------------------------------------------
// 2025-11-25 behaviour
// ---------------------------------------------------------------------------

Deno.test("2025-11-25: tool failures are isError results, unknown tools stay protocol errors", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(handler()).serveConnection(t);
  t.push(init(1, "2025-11-25"));
  await t.response(1);

  t.push(call(2, "fail"));
  const r2 = await t.response(2);
  assertEquals(r2.result, {
    content: [{ type: "text", text: "disk on fire" }],
    isError: true,
  });
  // SEP-1303: input validation errors are tool execution errors too.
  t.push(call(3, "fail", { kind: "params" }));
  const r3 = await t.response(3);
  assertEquals(r3.result.isError, true);
  assertEquals(r3.result.content[0].text, "path must be absolute");

  t.push(call(4, "fail", { kind: "notfound" }));
  assertEquals((await t.response(4)).error.code, -32001);
  t.end();
  await session.done;
});

Deno.test("before 2025-11-25 a thrown tool error is still a -32603 error", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(handler()).serveConnection(t);
  t.push(init(1, "2025-06-18"));
  await t.response(1);
  t.push(call(2, "fail"));
  const r = await t.response(2);
  assertEquals(r.error.code, -32603);
  assert(!("result" in r));
  t.end();
  await session.done;
});

Deno.test("tools/list passes optional Tool fields (title, icons, annotations) through", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(handler()).serveConnection(t);
  t.push(init(1));
  t.push({ jsonrpc: "2.0", id: 2, method: "tools/list" });
  const [whoami, fail] = (await t.response(2)).result.tools;
  assertEquals(whoami.title, "Who am I");
  assertEquals(whoami.icons[0].mimeType, "image/png");
  assertEquals(whoami.annotations, { readOnlyHint: true });
  assertEquals(Object.keys(fail).sort(), [
    "description",
    "inputSchema",
    "name",
  ]);
  t.end();
  await session.done;
});
