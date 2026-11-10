// AnthropicChatProvider options against a fake `fetch` (no network): extra
// headers and body fields, a key resolved per request, an injected fetch,
// message shaping in convertMessages, and stop_details kept in metadata.

import { assert, assertEquals } from "@std/assert";
import { ChatOptions, Message } from "@rullama/core";
import {
  ANTHROPIC_CONTENT_KEY,
  ANTHROPIC_STOP_DETAILS_KEY,
  AnthropicChatProvider,
  convertMessages,
} from "./anthropic.ts";
import { collect, withFetch } from "./test_util.ts";

// deno-lint-ignore no-explicit-any
type Call = { url: string; headers: Record<string, string>; body: any };

/** A fake fetch that records each call and answers with `reply(callIndex)`. */
function recorder(reply: (i: number) => Response) {
  const calls: Call[] = [];
  const fetchFn: typeof fetch = (input, init) => {
    calls.push({
      url: String(input),
      headers: { ...(init?.headers as Record<string, string>) },
      body: JSON.parse(String(init?.body)),
    });
    return Promise.resolve(reply(calls.length - 1));
  };
  return { calls, fetchFn };
}

const okReply = () =>
  Response.json({
    content: [{ type: "text", text: "ok" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 1, output_tokens: 1 },
  });

const okStream = () =>
  new Response(
    [
      { type: "message_start", message: { usage: { input_tokens: 1 } } },
      { type: "message_stop" },
    ].map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""),
  );

const FALLBACK = {
  extraHeaders: { "anthropic-beta": "server-side-fallback-2026-07-01" },
  extraBody: { fallbacks: "default" },
};

// ---------------------------------------------------------------------------
// Extra headers / body, fetch injection, key per request
// ---------------------------------------------------------------------------

Deno.test("Anthropic options - extraHeaders and extraBody go on chat and streamChat through the injected fetch", async () => {
  const { calls, fetchFn } = recorder((i) => i === 0 ? okReply() : okStream());
  const p = new AnthropicChatProvider({
    apiKey: "k",
    model: "claude-opus-5-5",
    ...FALLBACK,
    fetch: fetchFn,
  });
  // The global fetch must not be used.
  await withFetch(() => {
    throw new Error("global fetch called");
  }, async () => {
    await p.chat([Message.user("hi")], undefined, new ChatOptions());
    await collect(
      p.streamChat([Message.user("hi")], undefined, new ChatOptions()),
    );
  });
  assertEquals(calls.length, 2);
  for (const c of calls) {
    assertEquals(c.url, "https://api.anthropic.com/v1/messages");
    assertEquals(
      c.headers["anthropic-beta"],
      "server-side-fallback-2026-07-01",
    );
    assertEquals(c.headers["x-api-key"], "k");
    assertEquals(c.body.fallbacks, "default");
    assertEquals(c.body.model, "claude-opus-5-5");
  }
  assertEquals([calls[0].body.stream, calls[1].body.stream], [false, true]);
});

Deno.test("Anthropic options - per-call extras merge over the provider's; built fields and auth win", async () => {
  const { calls, fetchFn } = recorder(okReply);
  const p = new AnthropicChatProvider(
    "real-key",
    "claude-opus-5-5",
    undefined,
    undefined,
    {
      extraHeaders: {
        "x-api-key": "spoof",
        "x-a": "provider",
        "x-b": "provider",
      },
      extraBody: {
        model: "other",
        max_tokens: 1,
        a: "provider",
        b: "provider",
      },
      fetch: fetchFn,
    },
  );
  await p.chat(
    [Message.user("hi")],
    undefined,
    new ChatOptions({
      max_tokens: 100,
      extra_headers: { "x-b": "call", "anthropic-version": "spoof" },
      extra_body: { b: "call", messages: [] },
    }),
  );
  const { headers, body } = calls[0];
  assertEquals(headers["x-api-key"], "real-key");
  assertEquals(headers["anthropic-version"], "2023-06-01");
  assertEquals([headers["x-a"], headers["x-b"]], ["provider", "call"]);
  assertEquals([body.a, body.b], ["provider", "call"]);
  assertEquals(body.model, "claude-opus-5-5");
  assertEquals(body.max_tokens, 100);
  assertEquals(body.messages, [{
    role: "user",
    content: [{ type: "text", text: "hi" }],
  }]);
});

Deno.test("Anthropic options - a key function is resolved on every request", async () => {
  const { calls, fetchFn } = recorder(okReply);
  const keys = ["key-1", "key-2"];
  let asked = 0;
  const p = new AnthropicChatProvider({
    apiKey: () => Promise.resolve(keys[asked++]),
    model: "claude-opus-5-5",
    fetch: fetchFn,
  });
  await p.chat([Message.user("a")], undefined, new ChatOptions());
  await p.chat([Message.user("b")], undefined, new ChatOptions());
  assertEquals(asked, 2);
  assertEquals(calls.map((c) => c.headers["x-api-key"]), keys);

  // A synchronous function works too.
  const sync = new AnthropicChatProvider(
    () => "sync-key",
    "m",
    undefined,
    undefined,
    {
      fetch: fetchFn,
    },
  );
  await sync.chat([Message.user("c")], undefined, new ChatOptions());
  assertEquals(calls[2].headers["x-api-key"], "sync-key");
});

Deno.test("Anthropic options - withProviderName keeps key, extras and fetch", async () => {
  const { calls, fetchFn } = recorder(okReply);
  const p = new AnthropicChatProvider({
    apiKey: () => "fn-key",
    model: "claude-opus-5-5",
    baseUrl: "https://gateway.test/v1/messages",
    ...FALLBACK,
    fetch: fetchFn,
  }).withProviderName("gateway");
  assertEquals(p.name, "gateway");
  await p.chat([Message.user("hi")], undefined, new ChatOptions());
  assertEquals(calls[0].url, "https://gateway.test/v1/messages");
  assertEquals(calls[0].headers["x-api-key"], "fn-key");
  assertEquals(
    calls[0].headers["anthropic-beta"],
    FALLBACK.extraHeaders["anthropic-beta"],
  );
  assertEquals(calls[0].body.fallbacks, "default");
});

// ---------------------------------------------------------------------------
// convertMessages: empty text, merged user turns, stored replies untouched
// ---------------------------------------------------------------------------

Deno.test("convertMessages - skips empty text, drops empty messages, merges user/tool turns", () => {
  const stored = [
    { type: "thinking", thinking: "", signature: "s" },
    { type: "text", text: "" },
    { type: "tool_use", id: "t1", name: "a", input: {} },
    { type: "tool_use", id: "t2", name: "b", input: {} },
  ];
  const converted = convertMessages([
    Message.system("sys"),
    Message.user(""),
    Message.user("first"),
    new Message({
      role: "user",
      content: [{ type: "text", text: "" }, { type: "text", text: "second" }],
    }),
    new Message({
      role: "assistant",
      content: "",
      metadata: { [ANTHROPIC_CONTENT_KEY]: stored },
    }),
    Message.toolResult("t1", "one"),
    Message.toolResult("t2", "two"),
    Message.assistant(""),
    Message.assistant("a1"),
    Message.assistant("a2"),
    new Message({ role: "user", content: [] }),
  ]);
  assertEquals(converted, [
    {
      role: "user",
      content: [
        { type: "text", text: "first" },
        { type: "text", text: "second" },
      ],
    },
    // Sent exactly as stored: its empty text block is not touched.
    { role: "assistant", content: stored },
    {
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "t1", content: "one" },
        { type: "tool_result", tool_use_id: "t2", content: "two" },
      ],
    },
    // Assistant turns are not merged.
    { role: "assistant", content: [{ type: "text", text: "a1" }] },
    { role: "assistant", content: [{ type: "text", text: "a2" }] },
  ]);
});

Deno.test("convertMessages - a stored assistant reply is never merged with a neighbour", () => {
  const stored = [{ type: "text", text: "kept" }];
  const converted = convertMessages([
    Message.assistant("before"),
    new Message({
      role: "assistant",
      content: "kept",
      metadata: { [ANTHROPIC_CONTENT_KEY]: stored },
    }),
    Message.assistant("after"),
  ]);
  assertEquals(converted.length, 3);
  assertEquals(converted[1], { role: "assistant", content: stored });
});

// ---------------------------------------------------------------------------
// stop_details
// ---------------------------------------------------------------------------

Deno.test("Anthropic chat - stop_details land in metadata.anthropicStopDetails", async () => {
  const details = { category: "cyber", explanation: "declined" };
  const { fetchFn } = recorder(() =>
    Response.json({
      content: [],
      stop_reason: "refusal",
      stop_details: details,
      usage: { input_tokens: 1, output_tokens: 0 },
    })
  );
  const p = new AnthropicChatProvider({
    apiKey: "k",
    model: "m",
    fetch: fetchFn,
  });
  const reply = await p.chat(
    [Message.user("hi")],
    undefined,
    new ChatOptions(),
  );
  assertEquals(ANTHROPIC_STOP_DETAILS_KEY, "anthropicStopDetails");
  assertEquals(reply.finish_reason, "refusal");
  assertEquals(reply.message.metadata, {
    [ANTHROPIC_STOP_DETAILS_KEY]: details,
  });

  // Never sent back: the message has no stored blocks, and its empty content
  // drops it from the next request.
  assertEquals(convertMessages([reply.message]), []);

  // null stop_details add nothing.
  const { fetchFn: plain } = recorder(() =>
    Response.json({
      content: [{ type: "text", text: "ok" }],
      stop_reason: "end_turn",
      stop_details: null,
      usage: { input_tokens: 1, output_tokens: 1 },
    })
  );
  const ok = await new AnthropicChatProvider({
    apiKey: "k",
    model: "m",
    fetch: plain,
  })
    .chat([Message.user("hi")], undefined, new ChatOptions());
  assert(!(ANTHROPIC_STOP_DETAILS_KEY in ok.message.metadata));
});

Deno.test("Anthropic streamChat - stop_details from message_delta go in the metadata chunk", async () => {
  const details = { category: "cyber" };
  const { fetchFn } = recorder(() =>
    new Response(
      [
        { type: "message_start", message: { usage: { input_tokens: 2 } } },
        {
          type: "message_delta",
          delta: { stop_reason: "refusal", stop_details: details },
          usage: { output_tokens: 0 },
        },
        { type: "message_stop" },
      ].map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""),
    )
  );
  const p = new AnthropicChatProvider({
    apiKey: "k",
    model: "m",
    fetch: fetchFn,
  });
  const chunks = await collect(
    p.streamChat([Message.user("hi")], undefined, new ChatOptions()),
  );
  assertEquals(chunks.slice(-2), [
    { type: "metadata", metadata: { [ANTHROPIC_STOP_DETAILS_KEY]: details } },
    { type: "done" },
  ]);
});
