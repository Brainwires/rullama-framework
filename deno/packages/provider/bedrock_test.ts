import { assertEquals } from "@std/assert";
import { Message } from "@rullama/core";
import {
  bedrockInvokeUrl,
  BedrockProvider,
  bedrockStreamUrl,
  convertMessages,
  convertTools,
  getSystemMessage,
  hexEncode,
  parseBedrockResponse,
  sha256,
  toAmzDate,
  toDateStamp,
} from "./bedrock.ts";

// ---------------------------------------------------------------------------
// Provider name tests
// ---------------------------------------------------------------------------

Deno.test("BedrockProvider - default name is 'bedrock'", () => {
  const provider = new BedrockProvider(
    "us-east-1",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    { accessKeyId: "AKID", secretAccessKey: "SECRET" },
  );
  assertEquals(provider.name, "bedrock");
});

Deno.test("BedrockProvider - custom provider name", () => {
  const provider = new BedrockProvider(
    "us-east-1",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    { accessKeyId: "AKID", secretAccessKey: "SECRET" },
    "my-bedrock",
  );
  assertEquals(provider.name, "my-bedrock");
});

// ---------------------------------------------------------------------------
// URL building tests
// ---------------------------------------------------------------------------

Deno.test("bedrockInvokeUrl - correct format", () => {
  const url = bedrockInvokeUrl("us-east-1", "anthropic.claude-3-sonnet");
  assertEquals(
    url,
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet/invoke",
  );
});

Deno.test("bedrockStreamUrl - correct format", () => {
  const url = bedrockStreamUrl("us-west-2", "anthropic.claude-3-haiku");
  assertEquals(
    url,
    "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-haiku/invoke-with-response-stream",
  );
});

// ---------------------------------------------------------------------------
// convertMessages tests
// ---------------------------------------------------------------------------

Deno.test("convertMessages - text message", () => {
  const messages = [Message.user("Hello")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "user");
  assertEquals(converted[0].content.length, 1);
  assertEquals(converted[0].content[0].type, "text");
  assertEquals(converted[0].content[0].text, "Hello");
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

Deno.test("convertMessages - tool use blocks", () => {
  const messages = [
    new Message({
      role: "assistant",
      content: [
        { type: "text", text: "Let me search" },
        {
          type: "tool_use",
          id: "call_1",
          name: "search",
          input: { q: "test" },
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].content.length, 2);
  assertEquals(converted[0].content[1].type, "tool_use");
  assertEquals(converted[0].content[1].name, "search");
});

Deno.test("convertMessages - tool result blocks", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "call_1", content: "result" },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted[0].content.length, 1);
  assertEquals(converted[0].content[0].type, "tool_result");
  assertEquals(converted[0].content[0].tool_use_id, "call_1");
});

Deno.test("convertMessages - empty list", () => {
  const converted = convertMessages([]);
  assertEquals(converted.length, 0);
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
  assertEquals(converted[0].name, "test_tool");
  assertEquals(converted[0].description, "A test tool");
});

Deno.test("convertTools - empty list", () => {
  const converted = convertTools([]);
  assertEquals(converted.length, 0);
});

// ---------------------------------------------------------------------------
// getSystemMessage tests
// ---------------------------------------------------------------------------

Deno.test("getSystemMessage - found", () => {
  const messages = [
    Message.system("Be helpful"),
    Message.user("Hello"),
  ];
  assertEquals(getSystemMessage(messages), "Be helpful");
});

Deno.test("getSystemMessage - not found", () => {
  const messages = [Message.user("Hello")];
  assertEquals(getSystemMessage(messages), undefined);
});

// ---------------------------------------------------------------------------
// parseBedrockResponse tests
// ---------------------------------------------------------------------------

Deno.test("parseBedrockResponse - text response", () => {
  const response = {
    content: [{ type: "text", text: "Hello!" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 10, output_tokens: 5 },
  };
  const chatResponse = parseBedrockResponse(response);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(chatResponse.message.text(), "Hello!");
  assertEquals(chatResponse.usage.prompt_tokens, 10);
  assertEquals(chatResponse.usage.completion_tokens, 5);
  assertEquals(chatResponse.usage.total_tokens, 15);
  assertEquals(chatResponse.finish_reason, "end_turn");
});

Deno.test("parseBedrockResponse - multiple blocks", () => {
  const response = {
    content: [
      { type: "text", text: "Let me search" },
      { type: "tool_use", id: "call_1", name: "search", input: { q: "test" } },
    ],
    stop_reason: "tool_use",
    usage: { input_tokens: 20, output_tokens: 15 },
  };
  const chatResponse = parseBedrockResponse(response);
  if (typeof chatResponse.message.content !== "string") {
    assertEquals(chatResponse.message.content.length, 2);
    assertEquals(chatResponse.message.content[1].type, "tool_use");
  }
});

// ---------------------------------------------------------------------------
// SigV4 helper tests
// ---------------------------------------------------------------------------

Deno.test("hexEncode - encodes bytes correctly", () => {
  const bytes = new Uint8Array([0, 1, 15, 16, 255]);
  assertEquals(hexEncode(bytes), "00010f10ff");
});

Deno.test("sha256 - known hash", async () => {
  const hash = await sha256("");
  assertEquals(
    hash,
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  );
});

Deno.test("sha256 - non-empty string", async () => {
  const hash = await sha256("hello");
  assertEquals(
    hash,
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
  );
});

Deno.test("toAmzDate - formats correctly", () => {
  const date = new Date("2024-01-15T10:30:45.123Z");
  assertEquals(toAmzDate(date), "20240115T103045Z");
});

Deno.test("toDateStamp - formats correctly", () => {
  const date = new Date("2024-01-15T10:30:45.123Z");
  assertEquals(toDateStamp(date), "20240115");
});
