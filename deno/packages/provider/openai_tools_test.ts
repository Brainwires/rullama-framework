import { assert, assertEquals } from "@std/assert";
import { Message, objectSchema } from "@rullama/core";
import {
  convertMessages,
  convertStreamChunk,
  convertTools,
  OpenAiChatProvider,
  parseOpenAIResponse,
} from "./openai.ts";

Deno.test("OpenAI: tool_use blocks become tool_calls and tool_result blocks become role:tool messages", () => {
  const msgs = [
    Message.user("What's the weather?"),
    new Message({
      role: "assistant",
      content: [
        { type: "text", text: "Checking." },
        {
          type: "tool_use",
          id: "call_1",
          name: "weather",
          input: { city: "Oslo" },
        },
      ],
    }),
    new Message({
      role: "user",
      content: [{
        type: "tool_result",
        tool_use_id: "call_1",
        content: "12°C",
      }],
    }),
  ];
  const wire = convertMessages(msgs);
  assertEquals(wire.length, 3);
  assertEquals(wire[1].role, "assistant");
  assertEquals(wire[1].tool_calls?.[0].id, "call_1");
  assertEquals(wire[1].tool_calls?.[0].function.name, "weather");
  assertEquals(JSON.parse(wire[1].tool_calls![0].function.arguments!), {
    city: "Oslo",
  });
  assertEquals(wire[2].role, "tool");
  assertEquals(wire[2].tool_call_id, "call_1");
  assertEquals(wire[2].content, "12°C");
});

Deno.test("OpenAI: tool_calls in a response become tool_use blocks", () => {
  const resp = parseOpenAIResponse({
    choices: [{
      message: {
        content: null,
        tool_calls: [{
          id: "call_9",
          type: "function",
          function: { name: "weather", arguments: '{"city":"Oslo"}' },
        }],
      },
      finish_reason: "tool_calls",
    }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  });
  const blocks = resp.message.content;
  assert(Array.isArray(blocks));
  assertEquals(blocks[0], {
    type: "tool_use",
    id: "call_9",
    name: "weather",
    input: { city: "Oslo" },
  });
});

Deno.test("OpenAI: streamed tool-call fragments yield tool_use then tool_input_delta", () => {
  const first = convertStreamChunk({
    choices: [{
      delta: {
        tool_calls: [{
          index: 0,
          id: "call_1",
          type: "function",
          function: { name: "weather", arguments: "" },
        }],
      },
    }],
  } as never);
  assertEquals(first[0].type, "tool_use");
  const more = convertStreamChunk({
    choices: [{
      delta: {
        tool_calls: [{
          index: 0,
          type: "function",
          function: { arguments: '{"city":' },
        }],
      },
    }],
  } as never);
  assertEquals(more, [{
    type: "tool_input_delta",
    id: "0",
    partial_json: '{"city":',
  }]);
});

Deno.test("OpenAI: tool schema is a full JSON Schema object", () => {
  const [t] = convertTools([{
    name: "weather",
    description: "d",
    input_schema: objectSchema({ city: { type: "string" } }, ["city"]),
  }]);
  assertEquals(t.function.parameters, {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  });
});

Deno.test("OpenAI: reasoning models are detected and get max_completion_tokens", () => {
  for (
    const m of ["o1", "o1-mini", "o3-pro", "o4-mini", "gpt-5", "gpt-5-mini"]
  ) {
    assert(OpenAiChatProvider.isReasoningModel(m), m);
  }
  for (const m of ["gpt-4o", "gpt-4.1", "omni-moderation", "o10x"]) {
    assert(!OpenAiChatProvider.isReasoningModel(m), m);
  }
});
