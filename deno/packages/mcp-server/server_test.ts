import { assert, assertEquals } from "@std/assert";
import type { JsonRpcResponse } from "@rullama/mcp-client";
import type { McpHandler } from "./handler.ts";
import {
  McpServer,
  parseInitializeParams,
  type RequestContext,
} from "./server.ts";
import type { ServerTransport } from "./transport/traits.ts";
import { StdioServerTransport } from "./transport/stdio.ts";
import { AuthMiddleware, constantTimeEqual } from "./middleware/auth.ts";
import {
  MAX_RATE_LIMIT_BUCKETS,
  RateLimitMiddleware,
} from "./middleware/rate_limit.ts";

/** Scripted transport: hands out `lines` then EOF, records every response. */
class ScriptedTransport implements ServerTransport {
  responses: JsonRpcResponse[] = [];
  constructor(private lines: string[]) {}
  readRequest(): Promise<string | null> {
    return Promise.resolve(this.lines.shift() ?? null);
  }
  writeResponse(response: string): Promise<void> {
    this.responses.push(JSON.parse(response));
    return Promise.resolve();
  }
}

const handler: McpHandler = {
  serverInfo: () => ({ name: "t", version: "0" }),
  capabilities: () => ({ tools: {} }),
  listTools: () => [],
  callTool: () => Promise.resolve({ content: [{ type: "text", text: "ok" }] }),
};

const ctxFor = (name: string | null): RequestContext =>
  ({
    metadata: new Map<string, unknown>(),
    clientInfo: name === null ? null : { name, version: "1" },
  }) as unknown as RequestContext;

Deno.test("parseInitializeParams tolerates {} / missing clientInfo / garbage", () => {
  assertEquals(parseInitializeParams(undefined).clientInfo.name, "unknown");
  assertEquals(parseInitializeParams({}).clientInfo.name, "unknown");
  assertEquals(parseInitializeParams("x").clientInfo.version, "unknown");
  assertEquals(
    parseInitializeParams({ clientInfo: { name: "c", version: "1" } })
      .clientInfo,
    { name: "c", version: "1" },
  );
});

Deno.test("initialize with empty params does not crash, and notifications get no reply", async () => {
  const transport = new ScriptedTransport([
    JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: {} }),
    JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
    JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list" }),
  ]);
  const server = new McpServer(handler, transport);
  await server.run();
  assertEquals(transport.responses.map((r) => r.id), [1, 2]);
  assert(!("error" in transport.responses[0]));
});

Deno.test("stdio transport skips blank lines and stops at EOF", async () => {
  const enc = new TextEncoder();
  const input = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(enc.encode('\n\n{"a":1}\n'));
      c.close();
    },
  });
  const t = new StdioServerTransport(input);
  assertEquals(await t.readRequest(), '{"a":1}');
  assertEquals(await t.readRequest(), null);
});

Deno.test("auth: constant-time compare and the token is stripped from params", async () => {
  assert(constantTimeEqual("abc", "abc"));
  assert(!constantTimeEqual("abc", "abd"));
  assert(!constantTimeEqual("abc", "abcd"));
  const mw = new AuthMiddleware("s3cret");
  const ctx = ctxFor(null);
  const params: Record<string, unknown> = { name: "x", _auth_token: "s3cret" };
  const ok = await mw.processRequest(
    { jsonrpc: "2.0", id: 1, method: "tools/call", params },
    ctx,
  );
  assertEquals(ok.type, "continue");
  assert(!("_auth_token" in params), "token must be stripped from params");
  const stays = await mw.processRequest(
    {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { _auth_token: "nope" },
    },
    ctx,
  );
  assertEquals(stays.type, "continue", "the connection stays authenticated");
  const denied = await mw.processRequest(
    {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { _auth_token: "nope" },
    },
    ctxFor(null),
  );
  assertEquals(denied.type, "reject");
});

Deno.test("rate limit: keyed per client and bounded in bucket count", async () => {
  const mw = new RateLimitMiddleware(1);
  const call = (ctx: RequestContext, tool: string) =>
    mw.processRequest(
      { jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: tool } },
      ctx,
    );
  assertEquals((await call(ctxFor("a"), "t")).type, "continue");
  assertEquals((await call(ctxFor("a"), "t")).type, "reject", "a is limited");
  assertEquals(
    (await call(ctxFor("b"), "t")).type,
    "continue",
    "b has its own bucket",
  );
  for (let i = 0; i < MAX_RATE_LIMIT_BUCKETS + 50; i++) {
    await call(ctxFor("c"), `tool-${i}`);
  }
  // deno-lint-ignore no-explicit-any
  assert((mw as any).buckets.size <= MAX_RATE_LIMIT_BUCKETS);
});
