import { assertEquals, assertThrows } from "@std/assert";
import { Message } from "@rullama/core";
import {
  convertMessages,
  convertStreamChunk,
  convertTools,
  OpenAiChatProvider,
  parseOpenAIResponse,
} from "./openai.ts";

// ---------------------------------------------------------------------------
// Provider name tests
// ---------------------------------------------------------------------------

Deno.test("OpenAiChatProvider - default name is 'openai'", () => {
  const provider = new OpenAiChatProvider("key", "gpt-4");
  assertEquals(provider.name, "openai");
});

Deno.test("OpenAiChatProvider - withProviderName overrides name", () => {
  const provider = new OpenAiChatProvider("key", "gpt-4").withProviderName(
    "groq",
  );
  assertEquals(provider.name, "groq");
});

Deno.test("OpenAiChatProvider - isO1Model", () => {
  assertEquals(OpenAiChatProvider.isO1Model("o1-preview"), true);
  assertEquals(OpenAiChatProvider.isO1Model("o1-mini"), true);
  assertEquals(OpenAiChatProvider.isO1Model("o3-preview"), true);
  assertEquals(OpenAiChatProvider.isO1Model("gpt-4"), false);
  assertEquals(OpenAiChatProvider.isO1Model("gpt-4o"), false);
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

Deno.test("convertMessages - system message", () => {
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

Deno.test("convertMessages - preserves order", () => {
  const messages = [
    Message.system("System"),
    Message.user("User 1"),
    Message.assistant("Assistant"),
    Message.user("User 2"),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 4);
  assertEquals(converted[0].role, "system");
  assertEquals(converted[1].role, "user");
  assertEquals(converted[2].role, "assistant");
  assertEquals(converted[3].role, "user");
});

Deno.test("convertMessages - empty list", () => {
  const converted = convertMessages([]);
  assertEquals(converted.length, 0);
});

Deno.test("convertMessages - single text block simplified", () => {
  const messages = [
    new Message({
      role: "user",
      content: [{ type: "text", text: "Hello world" }],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].content, "Hello world");
});

Deno.test("convertMessages - multiple blocks as array", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        { type: "text", text: "First block" },
        { type: "text", text: "Second block" },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(Array.isArray(converted[0].content), true);
  if (Array.isArray(converted[0].content)) {
    assertEquals(converted[0].content.length, 2);
  }
});

Deno.test("convertMessages - with name", () => {
  const messages = [
    new Message({ role: "user", content: "Hello", name: "user_1" }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].name, "user_1");
});

Deno.test("convertMessages - image block", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        {
          type: "image",
          source: { type: "base64", media_type: "image/png", data: "abc123" },
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(Array.isArray(converted[0].content), true);
  if (Array.isArray(converted[0].content)) {
    assertEquals(converted[0].content[0].type, "image_url");
  }
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
        properties: { arg1: { type: "string" } },
        required: ["arg1"],
      },
    },
  ];
  const converted = convertTools(tools);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].type, "function");
  assertEquals(converted[0].function.name, "test_tool");
});

Deno.test("convertTools - empty list", () => {
  const converted = convertTools([]);
  assertEquals(converted.length, 0);
});

Deno.test("convertTools - multiple tools", () => {
  const tools = [
    { name: "tool1", description: "First", input_schema: { type: "object" } },
    { name: "tool2", description: "Second", input_schema: { type: "object" } },
  ];
  const converted = convertTools(tools);
  assertEquals(converted.length, 2);
  assertEquals(converted[0].function.name, "tool1");
  assertEquals(converted[1].function.name, "tool2");
});

// ---------------------------------------------------------------------------
// parseOpenAIResponse tests
// ---------------------------------------------------------------------------

Deno.test("parseOpenAIResponse - basic text", () => {
  const response = {
    choices: [{
      message: { content: "Hello!" },
      finish_reason: "stop",
    }],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
  const chatResponse = parseOpenAIResponse(response);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(chatResponse.message.text(), "Hello!");
  assertEquals(chatResponse.usage.prompt_tokens, 10);
  assertEquals(chatResponse.usage.completion_tokens, 5);
  assertEquals(chatResponse.usage.total_tokens, 15);
  assertEquals(chatResponse.finish_reason, "stop");
});

Deno.test("parseOpenAIResponse - no choices throws", () => {
  const response = {
    choices: [],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
  assertThrows(() => parseOpenAIResponse(response), Error, "No choices");
});

Deno.test("parseOpenAIResponse - null content", () => {
  const response = {
    choices: [{
      message: { content: null },
      finish_reason: "stop",
    }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };
  const chatResponse = parseOpenAIResponse(response);
  assertEquals(chatResponse.message.text(), "");
});

// ---------------------------------------------------------------------------
// convertStreamChunk tests
// ---------------------------------------------------------------------------

Deno.test("convertStreamChunk - text content", () => {
  const chunk = {
    choices: [{ delta: { content: "Hello" } }],
  };
  const converted = convertStreamChunk(chunk);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].type, "text");
  if (converted[0].type === "text") {
    assertEquals(converted[0].text, "Hello");
  }
});

Deno.test("convertStreamChunk - usage", () => {
  const chunk = {
    choices: [],
    usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
  };
  const converted = convertStreamChunk(chunk);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].type, "usage");
  if (converted[0].type === "usage") {
    assertEquals(converted[0].usage.prompt_tokens, 20);
    assertEquals(converted[0].usage.completion_tokens, 10);
    assertEquals(converted[0].usage.total_tokens, 30);
  }
});

Deno.test("convertStreamChunk - empty", () => {
  const chunk = { choices: [] };
  const converted = convertStreamChunk(chunk);
  assertEquals(converted.length, 0);
});

Deno.test("convertStreamChunk - tool calls", () => {
  const chunk = {
    choices: [{
      delta: {
        tool_calls: [{
          id: "call_123",
          type: "function",
          function: { name: "get_weather", arguments: '{"city":"London"}' },
        }],
      },
    }],
  };
  const converted = convertStreamChunk(chunk);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].type, "tool_use");
  if (converted[0].type === "tool_use") {
    assertEquals(converted[0].id, "call_123");
    assertEquals(converted[0].name, "get_weather");
  }
});
