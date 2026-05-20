import { assert, assertEquals } from "@std/assert";
import { Message } from "@brainwires/core";
import {
  BrainwiresRelayProvider,
  buildRequestParts,
  DEFAULT_BACKEND_URL,
  DEV_BACKEND_URL,
  getBackendFromApiKey,
  maxOutputTokensForModel,
  parseSseEvents,
} from "./brainwires_relay.ts";

Deno.test("provider name", () => {
  const p = new BrainwiresRelayProvider(
    "test-key",
    "http://localhost:3000",
    "claude-3-5-sonnet-20241022",
  );
  assertEquals(p.name, "brainwires");
});

Deno.test("max output tokens per model", () => {
  assertEquals(maxOutputTokensForModel("claude-3-5-sonnet-20241022"), 8192);
  assertEquals(maxOutputTokensForModel("gpt-5-mini"), 32768);
  assertEquals(maxOutputTokensForModel("claude-3-opus-20240229"), 4096);
  assertEquals(maxOutputTokensForModel("gemini-1.5-pro"), 8192);
  assertEquals(maxOutputTokensForModel("o1-preview"), 65536);
  assertEquals(maxOutputTokensForModel("unknown-model"), 8192);
});

Deno.test("backend from api key prefix", () => {
  assertEquals(getBackendFromApiKey("bw_dev_abc"), DEV_BACKEND_URL);
  assertEquals(getBackendFromApiKey("bw_prod_xyz"), DEFAULT_BACKEND_URL);
  assertEquals(getBackendFromApiKey("bw_test_xyz"), DEFAULT_BACKEND_URL);
});

Deno.test("getSystemMessage extracts system text", () => {
  const p = new BrainwiresRelayProvider("k", "u", "claude-3-5-sonnet-20241022");
  const msgs = [
    Message.system("You are a helpful assistant"),
    Message.user("Hello"),
  ];
  assertEquals(p.getSystemMessage(msgs), "You are a helpful assistant");
});

Deno.test("buildRequestParts: simple last-user message", () => {
  const msgs = [Message.user("hello")];
  const parts = buildRequestParts(msgs);
  assertEquals(parts.current_content, "hello");
  assertEquals(parts.conversation_history.length, 0);
  assertEquals(parts.function_call_output, null);
});

Deno.test("buildRequestParts: tool result threads response_id", () => {
  const assistant = new Message({
    role: "assistant",
    content: [
      { type: "tool_use", id: "call-1", name: "read_file", input: { path: "x.ts" } },
    ],
    metadata: { response_id: "resp-42" },
  });
  const toolResult = new Message({
    role: "tool",
    content: [
      { type: "tool_result", tool_use_id: "call-1", content: "file contents" },
    ],
  });
  const msgs = [Message.user("read x.ts"), assistant, toolResult];
  const parts = buildRequestParts(msgs);
  assertEquals(parts.current_content, "");
  assertEquals(parts.function_call_output, {
    call_id: "call-1",
    name: "read_file",
    output: "file contents",
  });
  assertEquals(parts.previous_response_id, "resp-42");
});

Deno.test("parseSseEvents yields event/data pairs", async () => {
  const payload =
    `event: delta\ndata: {"delta":"Hi"}\n\nevent: complete\ndata: {}\n\n`;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(payload));
      controller.close();
    },
  });
  const events: { eventType: string; data: string }[] = [];
  for await (const e of parseSseEvents(body)) events.push(e);
  assertEquals(events.length, 2);
  assertEquals(events[0].eventType, "delta");
  assert(events[0].data.includes("Hi"));
  assertEquals(events[1].eventType, "complete");
});
