import { assertEquals } from "@std/assert";
import { Message } from "@rullama/core";
import {
  buildRequestBody,
  convertUsage,
  messagesToInput,
  OpenAiResponsesProvider,
  responseToChat,
  streamEventToChunks,
  toolsToResponseTools,
} from "./openai_responses.ts";

// ---------------------------------------------------------------------------
// Provider name tests
// ---------------------------------------------------------------------------

Deno.test("OpenAiResponsesProvider - default name is 'openai-responses'", () => {
  const provider = new OpenAiResponsesProvider("key", "gpt-4o");
  assertEquals(provider.name, "openai-responses");
});

Deno.test("OpenAiResponsesProvider - withProviderName overrides name", () => {
  const provider = new OpenAiResponsesProvider("key", "gpt-4o")
    .withProviderName("custom");
  assertEquals(provider.name, "custom");
});

Deno.test("OpenAiResponsesProvider - lastResponseId initially undefined", () => {
  const provider = new OpenAiResponsesProvider("key", "gpt-4o");
  assertEquals(provider.getLastResponseId(), undefined);
});

// ---------------------------------------------------------------------------
// messagesToInput tests
// ---------------------------------------------------------------------------

Deno.test("messagesToInput - simple user message", () => {
  const messages = [Message.user("Hello")];
  const [items, system] = messagesToInput(messages);
  assertEquals(items.length, 1);
  assertEquals(items[0].type, "message");
  assertEquals(items[0].role, "user");
  assertEquals(items[0].content, "Hello");
  assertEquals(system, undefined);
});

Deno.test("messagesToInput - system message extracted as instructions", () => {
  const messages = [
    Message.system("You are helpful"),
    Message.user("Hello"),
  ];
  const [items, system] = messagesToInput(messages);
  assertEquals(items.length, 1);
  assertEquals(system, "You are helpful");
});

Deno.test("messagesToInput - assistant message", () => {
  const messages = [Message.assistant("I can help")];
  const [items, _system] = messagesToInput(messages);
  assertEquals(items.length, 1);
  assertEquals(items[0].role, "assistant");
});

Deno.test("messagesToInput - tool result message", () => {
  const messages = [
    new Message({
      role: "tool",
      content: "result text",
      name: "call_123",
    }),
  ];
  const [items, _system] = messagesToInput(messages);
  assertEquals(items.length, 1);
  assertEquals(items[0].type, "function_call_output");
  assertEquals(items[0].call_id, "call_123");
  assertEquals(items[0].output, "result text");
});

Deno.test("messagesToInput - tool result in blocks", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "call_1", content: "42" },
      ],
    }),
  ];
  const [items, _system] = messagesToInput(messages);
  assertEquals(items.length, 1);
  assertEquals(items[0].type, "function_call_output");
  assertEquals(items[0].call_id, "call_1");
});

Deno.test("messagesToInput - empty list", () => {
  const [items, system] = messagesToInput([]);
  assertEquals(items.length, 0);
  assertEquals(system, undefined);
});

// ---------------------------------------------------------------------------
// toolsToResponseTools tests
// ---------------------------------------------------------------------------

Deno.test("toolsToResponseTools - basic tool", () => {
  const tools = [
    {
      name: "search",
      description: "Search the web",
      input_schema: {
        type: "object",
        properties: { q: { type: "string" } },
        required: ["q"],
      },
    },
  ];
  const converted = toolsToResponseTools(tools);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].type, "function");
  assertEquals(converted[0].name, "search");
  assertEquals(converted[0].description, "Search the web");
});

Deno.test("toolsToResponseTools - empty list", () => {
  const converted = toolsToResponseTools([]);
  assertEquals(converted.length, 0);
});

// ---------------------------------------------------------------------------
// responseToChat tests
// ---------------------------------------------------------------------------

Deno.test("responseToChat - text response", () => {
  const resp = {
    id: "resp_123",
    output: [{
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "Hello!" }],
    }],
    usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
  };
  const chatResponse = responseToChat(resp);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(chatResponse.message.text(), "Hello!");
  assertEquals(chatResponse.usage.prompt_tokens, 10);
  assertEquals(chatResponse.usage.completion_tokens, 5);
  assertEquals(chatResponse.usage.total_tokens, 15);
});

Deno.test("responseToChat - function call response", () => {
  const resp = {
    id: "resp_456",
    output: [
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "Let me search" }],
      },
      {
        type: "function_call",
        name: "search",
        arguments: '{"q":"test"}',
        call_id: "call_1",
      },
    ],
    usage: { input_tokens: 20, output_tokens: 10 },
  };
  const chatResponse = responseToChat(resp);
  if (typeof chatResponse.message.content !== "string") {
    assertEquals(chatResponse.message.content.length, 2);
    assertEquals(chatResponse.message.content[1].type, "tool_use");
    if (chatResponse.message.content[1].type === "tool_use") {
      assertEquals(chatResponse.message.content[1].name, "search");
    }
  }
});

Deno.test("responseToChat - empty output", () => {
  const resp = {
    id: "resp_789",
    output: [],
  };
  const chatResponse = responseToChat(resp);
  assertEquals(chatResponse.message.text(), "");
});

Deno.test("responseToChat - refusal block", () => {
  const resp = {
    id: "resp_ref",
    output: [{
      type: "message",
      role: "assistant",
      content: [{ type: "refusal", refusal: "I cannot help with that" }],
    }],
  };
  const chatResponse = responseToChat(resp);
  assertEquals(chatResponse.message.text(), "I cannot help with that");
});

// ---------------------------------------------------------------------------
// convertUsage tests
// ---------------------------------------------------------------------------

Deno.test("convertUsage - with usage", () => {
  const usage = convertUsage({
    input_tokens: 100,
    output_tokens: 50,
    total_tokens: 150,
  });
  assertEquals(usage.prompt_tokens, 100);
  assertEquals(usage.completion_tokens, 50);
  assertEquals(usage.total_tokens, 150);
});

Deno.test("convertUsage - without total_tokens", () => {
  const usage = convertUsage({ input_tokens: 10, output_tokens: 5 });
  assertEquals(usage.total_tokens, 15);
});

Deno.test("convertUsage - undefined", () => {
  const usage = convertUsage(undefined);
  assertEquals(usage.prompt_tokens, 0);
  assertEquals(usage.completion_tokens, 0);
  assertEquals(usage.total_tokens, 0);
});

// ---------------------------------------------------------------------------
// streamEventToChunks tests
// ---------------------------------------------------------------------------

Deno.test("streamEventToChunks - text delta", () => {
  const chunks = streamEventToChunks({
    type: "response.output_text.delta",
    delta: "Hello",
    item_id: "msg_1",
    output_index: 0,
    content_index: 0,
  });
  assertEquals(chunks?.length, 1);
  assertEquals(chunks?.[0].type, "text");
  if (chunks?.[0].type === "text") {
    assertEquals(chunks[0].text, "Hello");
  }
});

Deno.test("streamEventToChunks - function call added", () => {
  const chunks = streamEventToChunks({
    type: "response.output_item.added",
    item: {
      type: "function_call",
      name: "search",
      call_id: "call_1",
      arguments: "",
    },
    output_index: 0,
  });
  assertEquals(chunks?.length, 1);
  assertEquals(chunks?.[0].type, "tool_use");
});

Deno.test("streamEventToChunks - function call arguments delta", () => {
  const chunks = streamEventToChunks({
    type: "response.function_call_arguments.delta",
    delta: '{"q":',
    item_id: "fc_1",
    output_index: 0,
  });
  assertEquals(chunks?.length, 1);
  assertEquals(chunks?.[0].type, "tool_input_delta");
  if (chunks?.[0].type === "tool_input_delta") {
    assertEquals(chunks[0].partial_json, '{"q":');
  }
});

Deno.test("streamEventToChunks - response completed", () => {
  const chunks = streamEventToChunks({
    type: "response.completed",
    response: {
      id: "resp_1",
      output: [],
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    },
  });
  assertEquals(chunks?.length, 2);
  assertEquals(chunks?.[0].type, "usage");
  assertEquals(chunks?.[1].type, "done");
});

Deno.test("streamEventToChunks - response failed", () => {
  const chunks = streamEventToChunks({
    type: "response.failed",
    response: { id: "resp_2", output: [] },
  });
  assertEquals(chunks?.length, 1);
  assertEquals(chunks?.[0].type, "done");
});

Deno.test("streamEventToChunks - unknown event returns undefined", () => {
  const chunks = streamEventToChunks({
    type: "response.created",
    response: { id: "resp_3", output: [] },
  });
  assertEquals(chunks, undefined);
});

// ---------------------------------------------------------------------------
// buildRequestBody tests
// ---------------------------------------------------------------------------

Deno.test("buildRequestBody - minimal", () => {
  const body = buildRequestBody(
    "gpt-4o",
    [{ type: "message", role: "user", content: "Hello" }],
    undefined,
    undefined,
    // deno-lint-ignore no-explicit-any
    { temperature: 0.7, max_tokens: 4096 } as any,
  );
  assertEquals(body.model, "gpt-4o");
  assertEquals(body.input.length, 1);
  assertEquals(body.instructions, undefined);
  assertEquals(body.tools, undefined);
});

Deno.test("buildRequestBody - with instructions and tools", () => {
  const tools = [{
    type: "function" as const,
    name: "search",
    description: "Search",
    parameters: {},
  }];
  const body = buildRequestBody(
    "gpt-4o",
    [{ type: "message", role: "user", content: "Hi" }],
    "Be helpful",
    tools,
    // deno-lint-ignore no-explicit-any
    { temperature: 0.5 } as any,
  );
  assertEquals(body.instructions, "Be helpful");
  assertEquals(body.tools.length, 1);
  assertEquals(body.tool_choice, "auto");
});

Deno.test("buildRequestBody - with previous response id", () => {
  const body = buildRequestBody(
    "gpt-4o",
    [{ type: "message", role: "user", content: "Hi" }],
    undefined,
    undefined,
    // deno-lint-ignore no-explicit-any
    {} as any,
    "resp_prev",
  );
  assertEquals(body.previous_response_id, "resp_prev");
});
