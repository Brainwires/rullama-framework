// AnthropicChatProvider against a fake `fetch` (no network): thinking blocks
// kept and sent back unchanged (chat and streamChat), sampling options sent
// only when the caller set them, and ChatOptions.signal passed to fetch.

import { assert, assertEquals, assertRejects } from "@std/assert";
import { ChatOptions, type ChatResponse, Message } from "@rullama/core";
import {
  ANTHROPIC_CONTENT_KEY,
  AnthropicChatProvider,
  convertMessages,
} from "./anthropic.ts";
import { buildAnthropicBody } from "./anthropic_format.ts";
import {
  abortOnFirstChunk,
  collect,
  hangingStream,
  withFetch,
} from "./test_util.ts";

const bodyOf = (init: RequestInit) => JSON.parse(String(init.body));

const provider = () => new AnthropicChatProvider("test-key", "claude-opus-5-5");

/** An SSE body of `events` (`event:` + `data:` lines, as the API sends). */
function sse(events: { type: string; [k: string]: unknown }[]): Response {
  const text = events.map((e) =>
    `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`
  ).join("");
  return new Response(text, {
    headers: { "content-type": "text/event-stream" },
  });
}

const THINKING_REPLY = [
  { type: "thinking", thinking: "Let me check.", signature: "sig-1" },
  { type: "redacted_thinking", data: "opaque-blob" },
  { type: "text", text: "Checking the weather." },
  {
    type: "tool_use",
    id: "toolu_1",
    name: "weather",
    input: { city: "Paris" },
  },
];

// ---------------------------------------------------------------------------
// 1. Thinking blocks: kept, and sent back exactly as they came
// ---------------------------------------------------------------------------

Deno.test("Anthropic chat - keeps raw blocks (thinking included) and sends them back unchanged", async () => {
  const bodies: Record<string, unknown>[] = [];
  const replies = [
    { content: THINKING_REPLY, stop_reason: "tool_use" },
    {
      content: [
        { type: "thinking", thinking: "", signature: "sig-2" },
        { type: "text", text: "It is sunny." },
      ],
      stop_reason: "end_turn",
    },
  ];
  await withFetch((init) => {
    bodies.push(bodyOf(init));
    const reply = replies[bodies.length - 1];
    return Response.json({
      ...reply,
      usage: { input_tokens: 10, output_tokens: 5 },
    });
  }, async () => {
    const p = provider();
    const history = [Message.user("Weather in Paris?")];
    const first: ChatResponse = await p.chat(
      history,
      undefined,
      new ChatOptions(),
    );

    // The loop sees text and the tool call; thinking stays in the metadata.
    assertEquals(first.message.content, [
      { type: "text", text: "Checking the weather." },
      {
        type: "tool_use",
        id: "toolu_1",
        name: "weather",
        input: { city: "Paris" },
      },
    ]);
    assertEquals(first.message.metadata, {
      [ANTHROPIC_CONTENT_KEY]: THINKING_REPLY,
    });
    assertEquals(ANTHROPIC_CONTENT_KEY, "anthropicContent");

    history.push(first.message, Message.toolResult("toolu_1", "sunny"));
    await p.chat(history, undefined, new ChatOptions());

    assertEquals(bodies[1].messages, [
      { role: "user", content: [{ type: "text", text: "Weather in Paris?" }] },
      { role: "assistant", content: THINKING_REPLY },
      {
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: "toolu_1",
          content: "sunny",
        }],
      },
    ]);
  });
});

Deno.test("convertMessages - stored blocks win over content; other messages convert as before", () => {
  const stored = [
    { type: "thinking", thinking: "hmm", signature: "s" },
    { type: "text", text: "Hi" },
  ];
  const converted = convertMessages([
    new Message({
      role: "assistant",
      content: "Hi",
      metadata: { [ANTHROPIC_CONTENT_KEY]: stored, other: 1 },
    }),
    // Only assistant messages are replayed from metadata.
    new Message({
      role: "user",
      content: "next",
      metadata: { [ANTHROPIC_CONTENT_KEY]: stored },
    }),
    // An empty list (or none) falls back to the content.
    new Message({
      role: "assistant",
      content: "plain",
      metadata: { [ANTHROPIC_CONTENT_KEY]: [] },
    }),
    new Message({
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t",
        content: "boom",
        is_error: true,
      }],
    }),
  ]);
  assertEquals(converted, [
    { role: "assistant", content: stored },
    { role: "user", content: [{ type: "text", text: "next" }] },
    { role: "assistant", content: [{ type: "text", text: "plain" }] },
    {
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "t",
        content: "boom",
        is_error: true,
      }],
    },
  ]);
});

Deno.test("Anthropic streamChat - gathers thinking, signature, text and tool input into the stored blocks", async () => {
  const bodies: Record<string, unknown>[] = [];
  await withFetch((init) => {
    bodies.push(bodyOf(init));
    return sse([
      { type: "message_start", message: { usage: { input_tokens: 12 } } },
      {
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "Let me " },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "check." },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "signature_delta", signature: "sig-1" },
      },
      { type: "content_block_stop", index: 0 },
      {
        type: "content_block_start",
        index: 1,
        content_block: { type: "redacted_thinking", data: "opaque-blob" },
      },
      { type: "content_block_stop", index: 1 },
      {
        type: "content_block_start",
        index: 2,
        content_block: { type: "text", text: "" },
      },
      {
        type: "content_block_delta",
        index: 2,
        delta: { type: "text_delta", text: "Checking " },
      },
      {
        type: "content_block_delta",
        index: 2,
        delta: { type: "text_delta", text: "the weather." },
      },
      { type: "content_block_stop", index: 2 },
      {
        type: "content_block_start",
        index: 3,
        content_block: {
          type: "tool_use",
          id: "toolu_1",
          name: "weather",
          input: {},
        },
      },
      {
        type: "content_block_delta",
        index: 3,
        delta: { type: "input_json_delta", partial_json: '{"city":' },
      },
      {
        type: "content_block_delta",
        index: 3,
        delta: { type: "input_json_delta", partial_json: '"Paris"}' },
      },
      { type: "content_block_stop", index: 3 },
      {
        type: "message_delta",
        delta: { stop_reason: "tool_use" },
        usage: { output_tokens: 30 },
      },
      { type: "message_stop" },
    ]);
  }, async () => {
    const chunks = await collect(
      provider().streamChat(
        [Message.user("Weather in Paris?")],
        undefined,
        new ChatOptions(),
      ),
    );
    assertEquals(chunks, [
      // Thinking is never streamed as text.
      { type: "text", text: "Checking " },
      { type: "text", text: "the weather." },
      { type: "tool_use", id: "toolu_1", name: "weather" },
      { type: "tool_input_delta", id: "toolu_1", partial_json: '{"city":' },
      { type: "tool_input_delta", id: "toolu_1", partial_json: '"Paris"}' },
      {
        type: "usage",
        usage: { prompt_tokens: 12, completion_tokens: 30, total_tokens: 42 },
      },
      {
        type: "metadata",
        metadata: { [ANTHROPIC_CONTENT_KEY]: THINKING_REPLY },
      },
      { type: "done" },
    ]);
    assertEquals(bodies[0].stream, true);
    assertEquals("temperature" in bodies[0], false);

    // Storing the metadata chunk on the assembled message replays it as-is.
    const meta = chunks.find((c) => c.type === "metadata");
    assert(meta?.type === "metadata");
    const assembled = new Message({
      role: "assistant",
      content: "Checking the weather.",
      metadata: meta.metadata,
    });
    assertEquals(convertMessages([assembled]), [
      { role: "assistant", content: THINKING_REPLY },
    ]);
  });
});

Deno.test("Anthropic streamChat - an error event mid-stream rejects", async () => {
  await withFetch(
    () =>
      sse([
        { type: "message_start", message: { usage: { input_tokens: 1 } } },
        {
          type: "error",
          error: { type: "overloaded_error", message: "Overloaded" },
        },
      ]),
    async () => {
      await assertRejects(
        () =>
          collect(
            provider().streamChat(
              [Message.user("hi")],
              undefined,
              new ChatOptions(),
            ),
          ),
        Error,
        "Overloaded",
      );
    },
  );
});

// ---------------------------------------------------------------------------
// 2. Sampling options only when the caller set them
// ---------------------------------------------------------------------------

Deno.test("Anthropic chat - default ChatOptions send no sampling options", async () => {
  let body: Record<string, unknown> = {};
  await withFetch((init) => {
    body = bodyOf(init);
    return Response.json({
      content: [{ type: "text", text: "ok" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 1, output_tokens: 1 },
    });
  }, async () => {
    const options = new ChatOptions();
    // The shared default is unchanged for other providers.
    assertEquals(options.temperature, 0.7);
    const reply = await provider().chat(
      [Message.user("hi")],
      undefined,
      options,
    );
    assertEquals(reply.message.content, "ok");
  });
  assertEquals("temperature" in body, false);
  assertEquals("top_p" in body, false);
  assertEquals("top_k" in body, false);
  assertEquals(body.max_tokens, 4096);
  assertEquals(body.model, "claude-opus-5-5");
});

Deno.test("buildAnthropicBody - sends sampling options the caller set", () => {
  const body = (options: ChatOptions | Partial<ChatOptions>) =>
    buildAnthropicBody(
      [Message.user("hi")],
      undefined,
      options as ChatOptions,
      {},
    );

  // Explicit values, including an explicit 0.7 and 0.
  assertEquals(body(new ChatOptions({ temperature: 0.7 })).temperature, 0.7);
  assertEquals(body(new ChatOptions({ temperature: 0 })).temperature, 0);
  assertEquals(body(ChatOptions.create().setTemperature(0.7)).temperature, 0.7);
  assertEquals(body(ChatOptions.deterministic(10)).temperature, 0);

  // A value assigned over the default counts as explicit.
  const assigned = new ChatOptions();
  assigned.temperature = 0.2;
  assertEquals(body(assigned).temperature, 0.2);

  // Copying defaulted options keeps them defaulted.
  assertEquals(
    "temperature" in body(new ChatOptions(new ChatOptions())),
    false,
  );

  // Plain objects: whatever they carry is explicit; nothing means nothing.
  assertEquals(body({ temperature: 0.7 }).temperature, 0.7);
  assertEquals("temperature" in body({ max_tokens: 5 }), false);

  const all = body(
    new ChatOptions({ temperature: 0.5, top_p: 0.9, top_k: 40 }),
  );
  assertEquals([all.temperature, all.top_p, all.top_k], [0.5, 0.9, 40]);
  assertEquals(body(ChatOptions.create().setTopK(5)).top_k, 5);
});

// ---------------------------------------------------------------------------
// 3. ChatOptions.signal reaches fetch (chat and streamChat)
// ---------------------------------------------------------------------------

Deno.test("Anthropic chat - passes ChatOptions.signal to fetch", async () => {
  const controller = new AbortController();
  controller.abort();
  let sent: AbortSignal | null | undefined;
  await withFetch((init) => {
    sent = init.signal;
    throw new DOMException("aborted", "AbortError");
  }, async () => {
    await assertRejects(
      () =>
        provider().chat(
          [Message.user("hi")],
          undefined,
          new ChatOptions({ signal: controller.signal }),
        ),
      DOMException,
    );
  });
  assertEquals(sent, controller.signal);
});

Deno.test("Anthropic streamChat - ChatOptions.signal aborts mid-stream", async () => {
  const events: string[] = [];
  let sent: AbortSignal | null | undefined;
  const first = [
    { type: "message_start", message: { usage: { input_tokens: 1 } } },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "Hel" },
    },
  ].map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  await withFetch((init) => {
    sent = init.signal;
    return hangingStream(init, events, first);
  }, async () => {
    const { got, signal } = await abortOnFirstChunk(provider());
    assertEquals(got, [{ type: "text", text: "Hel" }]);
    assertEquals(sent, signal);
  });
  assertEquals(events, ["aborted"]);
  assert(sent?.aborted);
});
