import { assertEquals } from "@std/assert";
import { Message } from "@rullama/core";
import {
  AnthropicChatProvider,
  convertMessages,
  convertTools,
  getSystemMessage,
  parseAnthropicResponse,
} from "./anthropic.ts";

// ---------------------------------------------------------------------------
// Provider name tests
// ---------------------------------------------------------------------------

Deno.test("AnthropicChatProvider - default name is 'anthropic'", () => {
  const provider = new AnthropicChatProvider("test-key", "claude-3-sonnet");
  assertEquals(provider.name, "anthropic");
});

Deno.test("AnthropicChatProvider - withProviderName overrides name", () => {
  const provider = new AnthropicChatProvider("test-key", "claude-3-sonnet")
    .withProviderName("bedrock");
  assertEquals(provider.name, "bedrock");
});

// ---------------------------------------------------------------------------
// convertMessages tests
// ---------------------------------------------------------------------------

Deno.test("convertMessages - text message", () => {
  const messages = [Message.user("Hello")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "user");
  assertEquals(converted[0].content, [{ type: "text", text: "Hello" }]);
});

Deno.test("convertMessages - filters system messages", () => {
  const messages = [
    Message.system("System prompt"),
    Message.user("Hello"),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "user");
});

Deno.test("convertMessages - assistant role", () => {
  const messages = [Message.assistant("I can help")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "assistant");
});

Deno.test("convertMessages - with content blocks", () => {
  const messages = [
    new Message({
      role: "assistant",
      content: [
        { type: "text", text: "Response" },
        {
          type: "tool_use",
          id: "tool-1",
          name: "test_tool",
          input: { arg: "value" },
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "assistant");
  assertEquals(converted[0].content.length, 2);
});

Deno.test("convertMessages - with tool result", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "tool-1", content: "Result" },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].content.length, 1);
  assertEquals(converted[0].content[0].type, "tool_result");
});

Deno.test("convertMessages - empty list", () => {
  const converted = convertMessages([]);
  assertEquals(converted.length, 0);
});

Deno.test("convertMessages - multiple roles", () => {
  const messages = [
    Message.user("Question"),
    Message.assistant("Answer"),
    Message.user("Follow-up"),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 3);
  assertEquals(converted[0].role, "user");
  assertEquals(converted[1].role, "assistant");
  assertEquals(converted[2].role, "user");
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
  assertEquals(converted[0].name, "test_tool");
  assertEquals(converted[0].description, "A test tool");
  assertEquals(converted[0].input_schema.arg1.type, "string");
});

Deno.test("convertTools - empty list", () => {
  const converted = convertTools([]);
  assertEquals(converted.length, 0);
});

Deno.test("convertTools - multiple tools", () => {
  const tools = [
    {
      name: "tool1",
      description: "First tool",
      input_schema: { type: "object" },
    },
    {
      name: "tool2",
      description: "Second tool",
      input_schema: { type: "object" },
    },
  ];
  const converted = convertTools(tools);
  assertEquals(converted.length, 2);
  assertEquals(converted[0].name, "tool1");
  assertEquals(converted[1].name, "tool2");
});

// ---------------------------------------------------------------------------
// getSystemMessage tests
// ---------------------------------------------------------------------------

Deno.test("getSystemMessage - found", () => {
  const messages = [
    Message.system("You are a helpful assistant"),
    Message.user("Hello"),
  ];
  const system = getSystemMessage(messages);
  assertEquals(system, "You are a helpful assistant");
});

Deno.test("getSystemMessage - not found", () => {
  const messages = [Message.user("Hello")];
  const system = getSystemMessage(messages);
  assertEquals(system, undefined);
});

// ---------------------------------------------------------------------------
// parseAnthropicResponse tests
// ---------------------------------------------------------------------------

Deno.test("parseAnthropicResponse - single text", () => {
  const response = {
    content: [{ type: "text" as const, text: "Hello!" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 10, output_tokens: 5 },
  };
  const chatResponse = parseAnthropicResponse(response);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(chatResponse.message.text(), "Hello!");
  assertEquals(chatResponse.usage.prompt_tokens, 10);
  assertEquals(chatResponse.usage.completion_tokens, 5);
  assertEquals(chatResponse.usage.total_tokens, 15);
  assertEquals(chatResponse.finish_reason, "end_turn");
});

Deno.test("parseAnthropicResponse - multiple blocks", () => {
  const response = {
    content: [
      { type: "text" as const, text: "Let me search" },
      {
        type: "tool_use" as const,
        id: "call_1",
        name: "search",
        input: { q: "test" },
      },
    ],
    stop_reason: "tool_use",
    usage: { input_tokens: 20, output_tokens: 15 },
  };
  const chatResponse = parseAnthropicResponse(response);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(typeof chatResponse.message.content !== "string", true);
  if (typeof chatResponse.message.content !== "string") {
    assertEquals(chatResponse.message.content.length, 2);
  }
});
