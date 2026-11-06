import { assert, assertEquals } from "@std/assert";
import type { CallToolResult } from "@rullama/mcp-client";
import type { McpHandler } from "./handler.ts";
import {
  McpServer,
  negotiateProtocolVersion,
  type RequestContext,
} from "./server.ts";
import { AuthMiddleware } from "./middleware/auth.ts";
import { delay, MemoryTransport, type Msg } from "./test_util.ts";

const text = (t: string): CallToolResult => ({
  content: [{ type: "text", text: t }],
});

/** A handler with a slow tool (waits for its signal or `ms`) and a fast one. */
function testHandler(log: string[] = []): McpHandler & {
  aborted: string[];
  shutdowns: number;
} {
  const h = {
    aborted: [] as string[],
    shutdowns: 0,
    serverInfo: () => ({ name: "t", version: "0" }),
    capabilities: () => ({ tools: { listChanged: true }, logging: {} }),
    listTools: () => [
      { name: "slow", description: "", inputSchema: { type: "object" } },
      { name: "fast", description: "", inputSchema: { type: "object" } },
    ],
    callTool: (
      name: string,
      args: Record<string, unknown>,
      ctx: RequestContext,
    ): Promise<CallToolResult> => {
      log.push(`start ${name}`);
      if (name === "fast") return Promise.resolve(text(`fast ${args.n ?? ""}`));
      const ms = typeof args.ms === "number" ? args.ms : 60_000;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => resolve(text(`slow ${ms}`)), ms);
        ctx.signal.addEventListener("abort", () => {
          clearTimeout(timer);
          h.aborted.push(String(ctx.requestId));
          reject(ctx.signal.reason);
        }, { once: true });
      });
    },
    onShutdown: () => {
      h.shutdowns += 1;
      return Promise.resolve();
    },
  };
  return h;
}

const init = (id: number, protocolVersion?: string): Msg => ({
  jsonrpc: "2.0",
  id,
  method: "initialize",
  params: {
    ...(protocolVersion ? { protocolVersion } : {}),
    capabilities: {},
    clientInfo: { name: "c", version: "1" },
  },
});

const call = (id: number | string, name: string, args: Msg = {}): Msg => ({
  jsonrpc: "2.0",
  id,
  method: "tools/call",
  params: { name, arguments: args },
});

/** A session over a fresh transport, past `initialize`. */
async function initialized(handler: McpHandler) {
  const t = new MemoryTransport();
  const session = new McpServer(handler).serveConnection(t);
  t.push(init(1));
  await t.response(1);
  return { t, session };
}

Deno.test("negotiateProtocolVersion echoes a supported version, else the latest", () => {
  assertEquals(negotiateProtocolVersion("2025-06-18"), "2025-06-18");
  assertEquals(negotiateProtocolVersion("2025-03-26"), "2025-03-26");
  assertEquals(negotiateProtocolVersion("2024-11-05"), "2024-11-05");
  assertEquals(negotiateProtocolVersion("2099-01-01"), "2025-06-18");
  assertEquals(negotiateProtocolVersion("x", ["b", "a"]), "b");
});

Deno.test("initialize replies with the negotiated protocol version", async () => {
  for (
    const [asked, expected] of [
      ["2025-06-18", "2025-06-18"],
      ["2025-03-26", "2025-03-26"],
      ["2024-11-05", "2024-11-05"],
      ["1999-01-01", "2025-06-18"],
    ]
  ) {
    const t = new MemoryTransport();
    const session = new McpServer(testHandler()).serveConnection(t);
    t.push(init(1, asked));
    const r = await t.response(1);
    assertEquals(r.result.protocolVersion, expected, `asked ${asked}`);
    assertEquals(session.protocolVersion, expected);
    t.end();
    await session.done;
  }
});

Deno.test("a slow tools/call does not block tools/list or a fast call", async () => {
  const log: string[] = [];
  const { t, session } = await initialized(testHandler(log));
  t.push(call(2, "slow", { ms: 60_000 }));
  t.push({ jsonrpc: "2.0", id: 3, method: "tools/list" });
  t.push(call(4, "fast", { n: 1 }));

  const list = await t.response(3, 500);
  assertEquals(list.result.tools.length, 2);
  const fast = await t.response(4, 500);
  assertEquals(fast.result.content[0].text, "fast 1");
  assertEquals(session.inFlight, 1, "slow call still running");
  assert(!t.written.some((m) => m.id === 2));

  await session.close(); // aborts the slow call
  assertEquals(session.inFlight, 0);
});

Deno.test("responses come back out of order and are matched by id", async () => {
  const { t, session } = await initialized(testHandler());
  t.push(call("a", "slow", { ms: 80 }));
  t.push(call("b", "slow", { ms: 10 }));
  t.push(call("c", "fast", { n: 3 }));
  await t.response("a");
  const order = t.written.filter((m) => typeof m.id === "string").map((m) =>
    m.id
  );
  assertEquals(order, ["c", "b", "a"]);
  assertEquals(
    t.written.find((m) => m.id === "a")!.result.content[0].text,
    "slow 80",
  );
  assertEquals(
    t.written.find((m) => m.id === "b")!.result.content[0].text,
    "slow 10",
  );
  t.end();
  await session.done;
});

Deno.test("notifications/cancelled aborts ctx.signal and suppresses the response", async () => {
  const h = testHandler();
  const { t, session } = await initialized(h);
  t.push(call(7, "slow"));
  await delay(10);
  assertEquals(session.inFlight, 1);
  t.push({
    jsonrpc: "2.0",
    method: "notifications/cancelled",
    params: { requestId: 7, reason: "user hit esc" },
  });
  t.push(call(8, "fast"));
  await t.response(8);
  await delay(10);
  assertEquals(h.aborted, ["7"]);
  assertEquals(session.inFlight, 0);
  assert(!t.written.some((m) => m.id === 7), "no response for cancelled id");
  // Cancelling an unknown / finished id is ignored.
  t.push({
    jsonrpc: "2.0",
    method: "notifications/cancelled",
    params: { requestId: 99 },
  });
  t.end();
  await session.done;
  assertEquals(h.shutdowns, 1);
});

Deno.test("ping is answered with an empty result, even before initialize and past auth", async () => {
  const t = new MemoryTransport();
  const server = new McpServer(testHandler()).withMiddleware(
    new AuthMiddleware("secret"),
  );
  const session = server.serveConnection(t);
  t.push({ jsonrpc: "2.0", id: 1, method: "ping" });
  assertEquals((await t.response(1)).result, {});
  // Auth rejects a request, but never answers a notification.
  t.push({ jsonrpc: "2.0", method: "notifications/initialized" });
  t.push({ jsonrpc: "2.0", id: 2, method: "tools/list" });
  assertEquals((await t.response(2)).error.code, -32003);
  t.end();
  await session.done;
  assertEquals(t.written.map((m) => m.id), [1, 2]);
});

Deno.test("EOF drains in-flight requests by default; abortOnEof drops them", async () => {
  const t1 = new MemoryTransport();
  const s1 = new McpServer(testHandler()).serveConnection(t1);
  t1.push(call(1, "slow", { ms: 30 }));
  t1.end();
  await s1.done;
  assertEquals(t1.written.map((m) => m.id), [1]);

  const h = testHandler();
  const t2 = new MemoryTransport();
  const s2 = new McpServer(h).serveConnection(t2, { abortOnEof: true });
  t2.push(call(1, "slow"));
  await delay(5);
  t2.end();
  await s2.done;
  assertEquals(t2.written, []);
  assertEquals(h.aborted, ["1"]);
  assertEquals(h.shutdowns, 1);
});

Deno.test("server-initiated notifications: tools/list_changed and notifications/message", async () => {
  const { t, session } = await initialized(testHandler());
  assert(await session.notifyToolsListChanged());
  assert(await session.sendLogMessage("info", { hello: 1 }, "qutty"));
  t.push({
    jsonrpc: "2.0",
    id: 2,
    method: "logging/setLevel",
    params: { level: "warning" },
  });
  assertEquals((await t.response(2)).result, {});
  assertEquals(await session.sendLogMessage("info", "dropped"), false);
  assert(await session.sendLogMessage("error", "kept"));
  t.end();
  await session.done;
  assertEquals(await session.notify("x/late"), false, "closed session");

  const notes = t.written.filter((m) => "method" in m);
  assertEquals(notes, [
    { jsonrpc: "2.0", method: "notifications/tools/list_changed" },
    {
      jsonrpc: "2.0",
      method: "notifications/message",
      params: { level: "info", logger: "qutty", data: { hello: 1 } },
    },
    {
      jsonrpc: "2.0",
      method: "notifications/message",
      params: { level: "error", data: "kept" },
    },
  ]);
  assert(notes.every((m) => !("id" in m)));
});

Deno.test("ctx carries the connection identity, per-request id and shared metadata", async () => {
  const seen: Msg[] = [];
  const handler: McpHandler = {
    ...testHandler(),
    callTool: (_name, _args, ctx) => {
      ctx.metadata.set(
        "calls",
        ((ctx.metadata.get("calls") as number) ?? 0) + 1,
      );
      seen.push({
        id: ctx.requestId,
        conn: ctx.connection?.id,
        identity: ctx.connection?.identity,
        client: ctx.clientInfo?.name,
        version: ctx.protocolVersion,
      });
      return Promise.resolve(text("ok"));
    },
  };
  const t = new MemoryTransport();
  const session = new McpServer(handler).serveConnection(t, {
    connection: { id: "tile-7", transport: "unix", identity: { tile: 7 } },
  });
  t.push(init(1, "2025-03-26"));
  t.push(call(2, "x"));
  t.push(call(3, "x"));
  await t.response(3);
  await t.response(2);
  assertEquals(session.metadata.get("calls"), 2);
  assertEquals(seen.map((s) => s.id).sort(), [2, 3]);
  for (const s of seen) {
    assertEquals(s.conn, "tile-7");
    assertEquals(s.identity, { tile: 7 });
    assertEquals(s.client, "c");
    assertEquals(s.version, "2025-03-26");
  }
  t.end();
  await session.done;
});

Deno.test("malformed input: parse error and invalid request get id:null errors; client responses are ignored", async () => {
  const t = new MemoryTransport();
  const session = new McpServer(testHandler()).serveConnection(t);
  t.push("{not json");
  t.push("[1,2]");
  t.push({ jsonrpc: "2.0", id: 5, result: {} });
  t.push({ jsonrpc: "2.0", id: 6 });
  t.end();
  await session.done;
  assertEquals(t.written.map((m) => [m.id, m.error.code]), [
    [null, -32700],
    [null, -32600],
    [6, -32600],
  ]);
});

Deno.test("a failed write aborts in-flight work and ends the session", async () => {
  const h = testHandler();
  const t = new MemoryTransport();
  const session = new McpServer(h).serveConnection(t);
  t.push(call(1, "slow"));
  await delay(5);
  t.closed = true; // writes now fail, reads still pending
  assertEquals(await session.notify("x/y"), false);
  await session.done;
  assertEquals(h.aborted, ["1"]);
});
