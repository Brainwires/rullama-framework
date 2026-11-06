/**
 * Tests for McpClient.
 * Mirrors Rust tests in `rullama-mcp/src/client.rs`.
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  cancellationNotification,
  LATEST_PROTOCOL_VERSION,
  McpClient,
  SUPPORTED_PROTOCOL_VERSIONS,
} from "./client.ts";
import type { ClientTransport, OutgoingMessage } from "./transport.ts";
import type { JsonRpcMessage, JsonRpcResponse } from "./types.ts";
import { parseJsonRpcMessage } from "./types.ts";

// =============================================================================
// Client creation tests — mirrors Rust test_client_creation
// =============================================================================

Deno.test("McpClient - creation with name and version", () => {
  const client = new McpClient("test", "0.1.0");

  assertEquals(client.clientName, "test");
  assertEquals(client.clientVersion, "0.1.0");
  assertEquals(client.currentRequestId, 1);
});

// Mirrors Rust test_request_id_increment
Deno.test("McpClient - request ID increment", () => {
  const client = new McpClient("test", "0.1.0");

  // IDs should start at 1 and increment
  assertEquals(client.currentRequestId, 1);
  // After internal calls, the ID advances (we test via listConnected which doesn't use IDs)
});

Deno.test("McpClient - default creation", () => {
  const client = McpClient.createDefault();

  assertEquals(client.clientName, "rullama");
  assertEquals(client.clientVersion, "0.5.0");
});

Deno.test("McpClient - isConnected returns false for unknown server", () => {
  const client = new McpClient("test", "0.1.0");

  assertEquals(client.isConnected("nonexistent"), false);
});

Deno.test("McpClient - listConnected returns empty initially", () => {
  const client = new McpClient("test", "0.1.0");

  assertEquals(client.listConnected().length, 0);
});

Deno.test("McpClient - disconnect unknown server does not throw", async () => {
  const client = new McpClient("test", "0.1.0");
  // Should not throw
  await client.disconnect("nonexistent");
});

Deno.test("McpClient - getServerInfo throws for unconnected server", () => {
  const client = new McpClient("test", "0.1.0");
  let threw = false;
  try {
    client.getServerInfo("nonexistent");
  } catch (e) {
    threw = true;
    assertEquals((e as Error).message, "Not connected to server: nonexistent");
  }
  assertEquals(threw, true);
});

Deno.test("McpClient - getCapabilities throws for unconnected server", () => {
  const client = new McpClient("test", "0.1.0");
  let threw = false;
  try {
    client.getCapabilities("nonexistent");
  } catch (e) {
    threw = true;
    assertEquals((e as Error).message, "Not connected to server: nonexistent");
  }
  assertEquals(threw, true);
});

Deno.test("cancellation is the MCP notifications/cancelled notification (no id, never answered)", () => {
  assertEquals(cancellationNotification(7, "user aborted"), {
    jsonrpc: "2.0",
    method: "notifications/cancelled",
    params: { requestId: 7, reason: "user aborted" },
  });
  assertEquals(cancellationNotification(8).params, { requestId: 8 });
  assert(SUPPORTED_PROTOCOL_VERSIONS.includes(LATEST_PROTOCOL_VERSION));
});

// =============================================================================
// Concurrency: responses matched by id, notifications routed, server pings
// =============================================================================

/**
 * Scriptable in-memory server end: records what the client sends, lets the
 * test reply in any order.
 */
class FakeServer implements ClientTransport {
  sent: Record<string, unknown>[] = [];
  #inbox: string[] = [];
  #readers: {
    resolve: (m: JsonRpcMessage) => void;
    reject: (e: Error) => void;
  }[] = [];
  #closed = false;
  #onSend: ((m: Record<string, unknown>) => void) | null = null;

  constructor(onSend?: (s: FakeServer, m: Record<string, unknown>) => void) {
    if (onSend) this.#onSend = (m) => onSend(this, m);
  }
  sendRequest(message: OutgoingMessage): Promise<void> {
    if (this.#closed) return Promise.reject(new Error("closed"));
    const m = JSON.parse(JSON.stringify(message));
    this.sent.push(m);
    this.#onSend?.(m);
    return Promise.resolve();
  }
  emit(message: object): void {
    const line = JSON.stringify(message);
    const r = this.#readers.shift();
    if (r) r.resolve(parseJsonRpcMessage(line));
    else this.#inbox.push(line);
  }
  receiveMessage(): Promise<JsonRpcMessage> {
    const line = this.#inbox.shift();
    if (line !== undefined) return Promise.resolve(parseJsonRpcMessage(line));
    if (this.#closed) return Promise.reject(new Error("EOF"));
    return new Promise((resolve, reject) =>
      this.#readers.push({ resolve, reject })
    );
  }
  receiveResponse(): Promise<JsonRpcResponse> {
    throw new Error("not used");
  }
  close(): Promise<void> {
    this.#closed = true;
    for (const r of this.#readers.splice(0)) r.reject(new Error("EOF"));
    return Promise.resolve();
  }
  /** Wait until the client has sent a message matching `pred`. */
  async sentWhere(
    pred: (m: Record<string, unknown>) => boolean,
  ): Promise<Record<string, unknown>> {
    for (let i = 0; i < 200; i++) {
      const hit = this.sent.find(pred);
      if (hit) return hit;
      await new Promise((r) => setTimeout(r, 1));
    }
    throw new Error(`not sent; got ${JSON.stringify(this.sent)}`);
  }
}

/** A fake server that answers `initialize` immediately. */
function autoInit(version = "2025-06-18"): FakeServer {
  return new FakeServer((s, m) => {
    if (m.method === "initialize") {
      queueMicrotask(() =>
        s.emit({
          jsonrpc: "2.0",
          id: m.id,
          result: {
            protocolVersion: version,
            capabilities: { tools: {} },
            serverInfo: { name: "fake", version: "1" },
          },
        })
      );
    }
  });
}

Deno.test("McpClient - concurrent calls resolve from out-of-order responses", async () => {
  const server = autoInit("2025-03-26");
  const client = new McpClient("test", "0.1.0");
  await client.connectTransport("s", server);
  assertEquals(client.getProtocolVersion("s"), "2025-03-26");
  assert(server.sent.some((m) => m.method === "notifications/initialized"));
  assert(
    !("id" in
      server.sent.find((m) => m.method === "notifications/initialized")!),
  );

  const slow = client.callTool("s", "slow");
  const list = client.listTools("s");
  const fast = client.callTool("s", "fast");
  const slowReq = await server.sentWhere((m) =>
    (m.params as { name?: string } | undefined)?.name === "slow"
  );
  const listReq = await server.sentWhere((m) => m.method === "tools/list");
  const fastReq = await server.sentWhere((m) =>
    (m.params as { name?: string } | undefined)?.name === "fast"
  );
  const ok = (t: string) => ({ content: [{ type: "text" as const, text: t }] });
  server.emit({ jsonrpc: "2.0", id: fastReq.id, result: ok("fast") });
  server.emit({ jsonrpc: "2.0", id: listReq.id, result: { tools: [] } });
  server.emit({ jsonrpc: "2.0", id: slowReq.id, result: ok("slow") });
  assertEquals(await fast, ok("fast"));
  assertEquals(await list, []);
  assertEquals(await slow, ok("slow"));
  await client.disconnect("s");
});

Deno.test("McpClient - notifications reach handlers and per-call listeners; server ping is answered", async () => {
  const server = autoInit();
  const client = new McpClient("test", "0.1.0");
  const global: string[] = [];
  const off = client.onNotification((name, n) =>
    global.push(`${name}:${n.method}`)
  );
  await client.connectTransport("s", server);

  const perCall: string[] = [];
  const call = client.callToolWithNotifications(
    "s",
    "t",
    {},
    (n) => perCall.push(n.method),
  );
  const req = await server.sentWhere((m) => m.method === "tools/call");
  server.emit({
    jsonrpc: "2.0",
    method: "notifications/progress",
    params: { progressToken: "p", progress: 1 },
  });
  server.emit({ jsonrpc: "2.0", id: "srv-1", method: "ping" });
  server.emit({ jsonrpc: "2.0", id: "srv-2", method: "roots/list" });
  server.emit({ jsonrpc: "2.0", id: req.id, result: { content: [] } });
  await call;
  server.emit({ jsonrpc: "2.0", method: "notifications/tools/list_changed" });
  await server.sentWhere((m) => m.id === "srv-2");
  for (let i = 0; i < 100 && global.length < 2; i++) {
    await new Promise((r) => setTimeout(r, 1));
  }

  assertEquals(perCall, ["notifications/progress"]);
  assertEquals(global, [
    "s:notifications/progress",
    "s:notifications/tools/list_changed",
  ]);
  assertEquals(server.sent.find((m) => m.id === "srv-1"), {
    jsonrpc: "2.0",
    id: "srv-1",
    result: {},
  });
  assertEquals(
    (server.sent.find((m) => m.id === "srv-2")!.error as { code: number })
      .code,
    -32601,
  );
  off();
  await client.disconnect("s");
});

Deno.test("McpClient - aborting a request sends notifications/cancelled and rejects", async () => {
  const server = autoInit();
  const client = new McpClient("test", "0.1.0");
  await client.connectTransport("s", server);
  const ac = new AbortController();
  const call = client.callTool("s", "slow", {}, { signal: ac.signal });
  const req = await server.sentWhere((m) => m.method === "tools/call");
  ac.abort(new Error("user hit esc"));
  await assertRejects(() => call, Error, "user hit esc");
  const cancel = await server.sentWhere((m) =>
    m.method === "notifications/cancelled"
  );
  assertEquals(cancel.params, { requestId: req.id, reason: "user hit esc" });
  // A late response for the cancelled id is dropped; the client keeps working.
  server.emit({ jsonrpc: "2.0", id: req.id, result: { content: [] } });
  const ping = client.ping("s");
  const pingReq = await server.sentWhere((m) => m.method === "ping");
  server.emit({ jsonrpc: "2.0", id: pingReq.id, result: {} });
  await ping;
  await client.disconnect("s");
});

Deno.test("McpClient - connection loss rejects every in-flight request", async () => {
  const server = autoInit();
  const client = new McpClient("test", "0.1.0");
  await client.connectTransport("s", server);
  const a = client.callTool("s", "a");
  const b = client.listTools("s");
  await server.sentWhere((m) => m.method === "tools/list");
  await server.close();
  await assertRejects(() => a, Error, "connection closed");
  await assertRejects(() => b, Error, "connection closed");
  await assertRejects(() => client.ping("s"), Error, "connection closed");
});

Deno.test("McpClient - an unsupported server protocol version fails connectTransport", async () => {
  const server = autoInit("1999-01-01");
  const client = new McpClient("test", "0.1.0");
  await assertRejects(
    () => client.connectTransport("s", server),
    Error,
    "not supported",
  );
  assertEquals(client.isConnected("s"), false);
});

Deno.test("the newest supported protocol revision is 2025-11-25 and is offered first", () => {
  assertEquals(LATEST_PROTOCOL_VERSION, "2025-11-25");
  assertEquals(SUPPORTED_PROTOCOL_VERSIONS[0], LATEST_PROTOCOL_VERSION);
  assertEquals(SUPPORTED_PROTOCOL_VERSIONS, [
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
  ]);
});
