import { assert, assertEquals } from "@std/assert";
import { ToolContext } from "@brainwires/core";
import {
  type SessionBroker,
  SessionId,
  type SessionMessage,
  type SessionSummary,
  type SpawnedSession,
  type SpawnRequest,
} from "./broker.ts";
import {
  CTX_METADATA_SESSION_ID,
  MAX_HISTORY_LIMIT,
  SessionsTool,
  TOOL_SESSIONS_HISTORY,
  TOOL_SESSIONS_LIST,
  TOOL_SESSIONS_SEND,
  TOOL_SESSIONS_SPAWN,
} from "./sessions_tool.ts";

const FIXED_TS = "2026-01-01T00:00:00Z";

class MockBroker implements SessionBroker {
  history_calls: Array<[SessionId, number | null]> = [];
  send_calls: Array<[SessionId, string]> = [];
  spawn_calls: Array<[SessionId, SpawnRequest]> = [];
  list_ret: SessionSummary[] = [];
  history_ret: SessionMessage[] = [];

  list(): Promise<SessionSummary[]> {
    return Promise.resolve(this.list_ret);
  }
  history(id: SessionId, limit: number | null): Promise<SessionMessage[]> {
    this.history_calls.push([id, limit]);
    return Promise.resolve(this.history_ret);
  }
  send(id: SessionId, text: string): Promise<void> {
    this.send_calls.push([id, text]);
    return Promise.resolve();
  }
  spawn(parent: SessionId, req: SpawnRequest): Promise<SpawnedSession> {
    this.spawn_calls.push([parent, req]);
    return Promise.resolve({ id: new SessionId("child-1"), first_reply: null });
  }
}

function ctxWithSession(session: string): ToolContext {
  const ctx = new ToolContext();
  ctx.metadata[CTX_METADATA_SESSION_ID] = session;
  return ctx;
}

Deno.test("list_tool_schema_shape", () => {
  const tools = SessionsTool.getTools();
  const list = tools.find((t) => t.name === TOOL_SESSIONS_LIST);
  assert(list, "list tool present");
  const required = list.input_schema.required ?? [];
  assertEquals(required.length, 0, `sessions_list must have no required inputs, got ${JSON.stringify(required)}`);
  assert(list.description.length > 0);
  const names = tools.map((t) => t.name);
  assert(names.includes(TOOL_SESSIONS_LIST));
  assert(names.includes(TOOL_SESSIONS_HISTORY));
  assert(names.includes(TOOL_SESSIONS_SEND));
  assert(names.includes(TOOL_SESSIONS_SPAWN));
});

Deno.test("history_tool_rejects_missing_session_id", async () => {
  const broker = new MockBroker();
  const tool = new SessionsTool(broker, new SessionId("self"));
  const ctx = ctxWithSession("self");
  const result = await tool.execute("call-1", TOOL_SESSIONS_HISTORY, {}, ctx);
  assert(result.is_error, `expected error, got ${result.content}`);
  assert(result.content.toLowerCase().includes("session_id"));
  assertEquals(broker.history_calls.length, 0, "broker must not be called for invalid input");
});

Deno.test("history_tool_clamps_limit", async () => {
  const broker = new MockBroker();
  broker.history_ret.push({ role: "user", content: "hi", timestamp: FIXED_TS });
  const tool = new SessionsTool(broker, new SessionId("self"));
  const ctx = ctxWithSession("self");
  const result = await tool.execute(
    "call-1",
    TOOL_SESSIONS_HISTORY,
    { session_id: "target", limit: 9999 },
    ctx,
  );
  assert(!result.is_error, `unexpected error: ${result.content}`);
  assertEquals(broker.history_calls.length, 1);
  assert(broker.history_calls[0][0].equals("target"));
  assertEquals(
    broker.history_calls[0][1],
    MAX_HISTORY_LIMIT,
    `limit must be clamped to MAX_HISTORY_LIMIT (${MAX_HISTORY_LIMIT})`,
  );
});

Deno.test("send_tool_self_send_rejected", async () => {
  const broker = new MockBroker();
  const tool = new SessionsTool(broker, new SessionId("me"));
  const ctx = ctxWithSession("me");
  const result = await tool.execute(
    "call-1",
    TOOL_SESSIONS_SEND,
    { session_id: "me", text: "hello" },
    ctx,
  );
  assert(result.is_error);
  assert(
    result.content.toLowerCase().includes("recurs"),
    `error should mention recursion, got: ${result.content}`,
  );
  assertEquals(broker.send_calls.length, 0);
});

Deno.test("send_tool_forwards_to_broker_when_distinct", async () => {
  const broker = new MockBroker();
  const tool = new SessionsTool(broker, new SessionId("me"));
  const ctx = ctxWithSession("me");
  const result = await tool.execute(
    "call-1",
    TOOL_SESSIONS_SEND,
    { session_id: "peer", text: "ping" },
    ctx,
  );
  assert(!result.is_error, `unexpected error: ${result.content}`);
  assertEquals(broker.send_calls.length, 1);
  assert(broker.send_calls[0][0].equals("peer"));
  assertEquals(broker.send_calls[0][1], "ping");
  assert(result.content.includes('"ok"'));
});

Deno.test("spawn_tool_passes_through", async () => {
  const broker = new MockBroker();
  const tool = new SessionsTool(broker, new SessionId("parent"));
  const ctx = ctxWithSession("parent");
  const input = {
    prompt: "research the openclaw parity gap",
    model: "claude-opus-4-7",
    system: "you are a research agent",
    tools: ["fetch_url", "query_codebase"],
    wait_for_first_reply: true,
    wait_timeout_secs: 30,
  };
  const result = await tool.execute("call-1", TOOL_SESSIONS_SPAWN, input, ctx);
  assert(!result.is_error, `unexpected error: ${result.content}`);
  assertEquals(broker.spawn_calls.length, 1);
  assert(broker.spawn_calls[0][0].equals("parent"));
  const req = broker.spawn_calls[0][1];
  assertEquals(req.prompt, "research the openclaw parity gap");
  assertEquals(req.model, "claude-opus-4-7");
  assertEquals(req.system, "you are a research agent");
  assertEquals(req.tools, ["fetch_url", "query_codebase"]);
  assertEquals(req.wait_for_first_reply, true);
  assertEquals(req.wait_timeout_secs, 30);
});

Deno.test("spawn_tool_errors_without_parent_session", async () => {
  const broker = new MockBroker();
  // No metadata, no current_session_id → parent unknown.
  const tool = new SessionsTool(broker, null);
  const ctx = new ToolContext();
  const result = await tool.execute(
    "call-1",
    TOOL_SESSIONS_SPAWN,
    { prompt: "x" },
    ctx,
  );
  assert(result.is_error);
  assert(
    result.content.includes("session") && result.content.toLowerCase().includes("caller"),
  );
  assertEquals(broker.spawn_calls.length, 0);
});

Deno.test("list_returns_json_array", async () => {
  const broker = new MockBroker();
  broker.list_ret.push({
    id: new SessionId("s1"),
    channel: "discord",
    peer: "alice",
    created_at: FIXED_TS,
    last_active: FIXED_TS,
    message_count: 3,
    parent: null,
  });
  const tool = new SessionsTool(broker, new SessionId("me"));
  const ctx = ctxWithSession("me");
  const result = await tool.execute("c1", TOOL_SESSIONS_LIST, {}, ctx);
  assert(!result.is_error);
  assert(result.content.startsWith("["));
  assert(result.content.includes('"s1"'));
  assert(result.content.includes('"discord"'));
});
