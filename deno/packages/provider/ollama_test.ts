// deno-lint-ignore-file no-explicit-any
import { assert, assertEquals, assertRejects } from "@std/assert";
import { ChatOptions, Message } from "@rullama/core";
import {
  convertMessages,
  convertTools,
  OllamaChatProvider,
  parseOllamaResponse,
} from "./ollama.ts";
import {
  abortOnFirstChunk,
  collect,
  hangingStream,
  withFetch,
} from "./test_util.ts";

// ---------------------------------------------------------------------------
// Provider name tests
// ---------------------------------------------------------------------------

Deno.test("OllamaChatProvider - name is 'ollama'", () => {
  const provider = new OllamaChatProvider("llama2");
  assertEquals(provider.name, "ollama");
});

// ---------------------------------------------------------------------------
// convertMessages tests
// ---------------------------------------------------------------------------

Deno.test("convertMessages - text message", () => {
  const messages = [Message.user("Hello")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "user");
  assertEquals(converted[0].content, "Hello");
});

Deno.test("convertMessages - system role", () => {
  const messages = [Message.system("You are helpful")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "system");
});

Deno.test("convertMessages - assistant role", () => {
  const messages = [Message.assistant("I can help")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "assistant");
});

Deno.test("convertMessages - blocks joined with newline", () => {
  const messages = [
    new Message({
      role: "assistant",
      content: [
        { type: "text", text: "First" },
        { type: "text", text: "Second" },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].content, "First\nSecond");
});

Deno.test("convertMessages - image blocks extracted", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        { type: "text", text: "What's in this image?" },
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: "iVBORw0KGgo=",
          },
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].content, "What's in this image?");
  assertEquals(converted[0].images !== undefined, true);
  assertEquals(converted[0].images!.length, 1);
  assertEquals(converted[0].images![0], "iVBORw0KGgo=");
});

Deno.test("convertMessages - multiple images", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        { type: "text", text: "Compare these" },
        {
          type: "image",
          source: { type: "base64", media_type: "image/png", data: "img1" },
        },
        {
          type: "image",
          source: { type: "base64", media_type: "image/jpeg", data: "img2" },
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].images!.length, 2);
});

Deno.test("convertMessages - text only no images", () => {
  const messages = [
    new Message({
      role: "user",
      content: [{ type: "text", text: "Just text" }],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].images, undefined);
});

Deno.test("convertMessages - empty list", () => {
  const converted = convertMessages([]);
  assertEquals(converted.length, 0);
});

Deno.test("convertMessages - preserves order", () => {
  const messages = [
    Message.system("First"),
    Message.user("Second"),
    Message.assistant("Third"),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 3);
  assertEquals(converted[0].content, "First");
  assertEquals(converted[1].content, "Second");
  assertEquals(converted[2].content, "Third");
});

Deno.test("convertMessages - all roles", () => {
  const messages = [
    Message.system("system"),
    Message.user("user"),
    Message.assistant("assistant"),
    new Message({ role: "tool", content: "tool" }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].role, "system");
  assertEquals(converted[1].role, "user");
  assertEquals(converted[2].role, "assistant");
  assertEquals(converted[3].role, "tool");
});

// ---------------------------------------------------------------------------
// convertTools tests
// ---------------------------------------------------------------------------

Deno.test("convertTools - basic tool", () => {
  const tools = [
    {
      name: "test_tool",
      description: "A test tool",
      input_schema: {
        type: "object",
        properties: { arg1: { type: "string", description: "First argument" } },
        required: ["arg1"],
      },
    },
  ];
  const converted = convertTools(tools);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].type, "function");
  assertEquals(converted[0].function.name, "test_tool");
  assertEquals(converted[0].function.description, "A test tool");
  assertEquals(converted[0].function.parameters.type, "object");
  assertEquals(converted[0].function.parameters.required, ["arg1"]);
});

Deno.test("convertTools - empty list", () => {
  const converted = convertTools([]);
  assertEquals(converted.length, 0);
});

Deno.test("convertTools - multiple tools keep order and get an empty required list", () => {
  const tools = [
    { name: "tool1", description: "First", input_schema: { type: "object" } },
    { name: "tool2", description: "Second", input_schema: { type: "object" } },
  ];
  assertEquals(
    convertTools(tools).map((
      t,
    ) => [t.function.name, t.function.parameters.required]),
    [["tool1", []], ["tool2", []]],
  );
});

// ---------------------------------------------------------------------------
// parseOllamaResponse tests
// ---------------------------------------------------------------------------

Deno.test("parseOllamaResponse - basic response", () => {
  const response = {
    message: { content: "Hello, how can I help?" },
    done_reason: "stop",
    prompt_eval_count: 10,
    eval_count: 20,
  };
  const chatResponse = parseOllamaResponse(response);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(chatResponse.message.text(), "Hello, how can I help?");
  assertEquals(chatResponse.usage.prompt_tokens, 10);
  assertEquals(chatResponse.usage.completion_tokens, 20);
  assertEquals(chatResponse.usage.total_tokens, 30);
  assertEquals(chatResponse.finish_reason, "stop");
});

Deno.test("parseOllamaResponse - missing optional fields", () => {
  const response = {
    message: { content: "Response" },
  };
  const chatResponse = parseOllamaResponse(response);
  assertEquals(chatResponse.message.text(), "Response");
  assertEquals(chatResponse.usage.prompt_tokens, 0);
  assertEquals(chatResponse.usage.completion_tokens, 0);
  assertEquals(chatResponse.finish_reason, "stop");
});

// ---------------------------------------------------------------------------
// Tool calls and cancellation, through a fake fetch
// ---------------------------------------------------------------------------

const WEATHER_TOOL = {
  name: "get_weather",
  description: "Weather for a city",
  input_schema: {
    type: "object" as const,
    properties: { city: { type: "string" } },
    required: ["city"],
  },
};

const ndjson = (...objs: unknown[]) =>
  new Response(objs.map((o) => JSON.stringify(o) + "\n").join(""));

Deno.test("parseOllamaResponse - tool_calls become tool_use blocks", () => {
  const r = parseOllamaResponse({
    message: {
      content: "",
      tool_calls: [
        { function: { name: "get_weather", arguments: { city: "Oslo" } } },
        { id: "abc", function: { name: "get_time", arguments: '{"tz":1}' } },
      ],
    },
    done_reason: "stop",
  });
  assertEquals(r.message.content, [
    {
      type: "tool_use",
      id: "call_0",
      name: "get_weather",
      input: { city: "Oslo" },
    },
    { type: "tool_use", id: "abc", name: "get_time", input: { tz: 1 } },
  ]);
  // Ollama says "stop"; agent loops need to see that tools are pending.
  assertEquals(r.finish_reason, "tool_calls");
});

Deno.test("parseOllamaResponse - text alongside a tool call is kept", () => {
  const r = parseOllamaResponse({
    message: {
      content: "Checking.",
      tool_calls: [{ function: { name: "get_weather", arguments: {} } }],
    },
  });
  assertEquals(r.message.content, [
    { type: "text", text: "Checking." },
    { type: "tool_use", id: "call_0", name: "get_weather", input: {} },
  ]);
});

Deno.test("convertMessages - tool round trip uses native tool_calls and role tool", () => {
  const converted = convertMessages([
    Message.user("Weather in Oslo?"),
    new Message({
      role: "assistant",
      content: [{
        type: "tool_use",
        id: "call_0",
        name: "get_weather",
        input: { city: "Oslo" },
      }],
    }),
    new Message({
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "call_0",
        content: "Sunny",
      }],
    }),
  ]);
  assertEquals(converted, [
    { role: "user", content: "Weather in Oslo?" },
    {
      role: "assistant",
      content: "",
      tool_calls: [{
        id: "call_0",
        function: { name: "get_weather", arguments: { city: "Oslo" } },
      }],
    },
    {
      role: "tool",
      content: "Sunny",
      tool_call_id: "call_0",
      tool_name: "get_weather",
    },
  ]);
});

Deno.test("OllamaChatProvider.chat - tool-call round trip over /api/chat", async () => {
  const bodies: any[] = [];
  const replies = [
    Response.json({
      message: {
        role: "assistant",
        content: "",
        tool_calls: [{
          function: { name: "get_weather", arguments: { city: "Oslo" } },
        }],
      },
      done: true,
      done_reason: "stop",
      prompt_eval_count: 5,
      eval_count: 3,
    }),
    Response.json({
      message: { role: "assistant", content: "Sunny in Oslo." },
      done: true,
      done_reason: "stop",
    }),
  ];
  await withFetch((init) => {
    bodies.push(JSON.parse(String(init.body)));
    return replies.shift()!;
  }, async () => {
    const provider = new OllamaChatProvider("qwen3", "http://ollama.test");
    const history = [Message.user("Weather in Oslo?")];
    const first = await provider.chat(
      history,
      [WEATHER_TOOL],
      new ChatOptions(),
    );
    assertEquals(first.finish_reason, "tool_calls");
    const [use] = first.message.content as any[];
    assertEquals(use.name, "get_weather");
    assertEquals(use.input, { city: "Oslo" });

    history.push(
      first.message,
      new Message({
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: use.id,
          content: "Sunny",
        }],
      }),
    );
    const second = await provider.chat(
      history,
      [WEATHER_TOOL],
      new ChatOptions(),
    );
    assertEquals(second.message.text(), "Sunny in Oslo.");
    assertEquals(second.finish_reason, "stop");
  });

  assertEquals(bodies[0].tools[0].function.name, "get_weather");
  assertEquals(bodies[1].messages.slice(1), [
    {
      role: "assistant",
      content: "",
      tool_calls: [{
        id: "call_0",
        function: { name: "get_weather", arguments: { city: "Oslo" } },
      }],
    },
    {
      role: "tool",
      content: "Sunny",
      tool_call_id: "call_0",
      tool_name: "get_weather",
    },
  ]);
});

Deno.test("OllamaChatProvider.streamChat - streamed tool calls", async () => {
  await withFetch(
    () =>
      ndjson(
        { message: { content: "Let me check." }, done: false },
        {
          message: {
            content: "",
            tool_calls: [
              {
                function: { name: "get_weather", arguments: { city: "Oslo" } },
              },
              {
                id: "t2",
                function: { name: "get_weather", arguments: { city: "Rome" } },
              },
            ],
          },
          done: false,
        },
        { done: true, prompt_eval_count: 4, eval_count: 6 },
      ),
    async () => {
      const provider = new OllamaChatProvider("qwen3", "http://ollama.test");
      const chunks = await collect(
        provider.streamChat(
          [Message.user("hi")],
          [WEATHER_TOOL],
          new ChatOptions(),
        ),
      );
      assertEquals(chunks, [
        { type: "text", text: "Let me check." },
        { type: "tool_use", id: "call_0", name: "get_weather" },
        {
          type: "tool_input_delta",
          id: "call_0",
          partial_json: '{"city":"Oslo"}',
        },
        { type: "tool_use", id: "t2", name: "get_weather" },
        { type: "tool_input_delta", id: "t2", partial_json: '{"city":"Rome"}' },
        {
          type: "usage",
          usage: { prompt_tokens: 4, completion_tokens: 6, total_tokens: 10 },
        },
        { type: "done" },
      ]);
    },
  );
});

/** A fake streaming reply that sends one chunk ("Hel"), then hangs until the
 * request signal aborts or the reader cancels it. */
const hangingOllama = (init: RequestInit, events: string[]) =>
  hangingStream(
    init,
    events,
    JSON.stringify({ message: { content: "Hel" }, done: false }) + "\n",
  );

Deno.test("OllamaChatProvider.streamChat - ChatOptions.signal aborts mid-stream", async () => {
  const events: string[] = [];
  let sentSignal: AbortSignal | null | undefined;
  await withFetch((init) => {
    sentSignal = init.signal;
    return hangingOllama(init, events);
  }, async () => {
    const { got, signal } = await abortOnFirstChunk(
      new OllamaChatProvider("qwen3", "http://ollama.test"),
    );
    assertEquals(got, [{ type: "text", text: "Hel" }]);
    assertEquals(sentSignal, signal);
  });
  assertEquals(events, ["aborted"]);
});

Deno.test("OllamaChatProvider.streamChat - stopping early cancels the response body", async () => {
  const events: string[] = [];
  await withFetch((init) => hangingOllama(init, events), async () => {
    const provider = new OllamaChatProvider("qwen3", "http://ollama.test");
    for await (
      const _ of provider.streamChat(
        [Message.user("hi")],
        undefined,
        new ChatOptions(),
      )
    ) break;
  });
  // Closing the stream is what makes Ollama stop generating.
  assertEquals(events, ["cancelled"]);
});

Deno.test("OllamaChatProvider.chat - passes ChatOptions.signal to fetch", async () => {
  const controller = new AbortController();
  controller.abort();
  await withFetch((init) => {
    assert(init.signal?.aborted);
    throw new DOMException("aborted", "AbortError");
  }, async () => {
    const provider = new OllamaChatProvider("qwen3", "http://ollama.test");
    await assertRejects(
      () =>
        provider.chat(
          [Message.user("hi")],
          undefined,
          new ChatOptions({ signal: controller.signal }),
        ),
      DOMException,
    );
  });
});
