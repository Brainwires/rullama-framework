import { assertEquals } from "@std/assert";
import { ChatOptions, Message } from "@rullama/core";
import {
  buildGeminiRequest,
  convertCandidateContent,
  convertMessages,
  convertTools,
  getSystemInstruction,
  GoogleChatProvider,
  parseGeminiResponse,
} from "./gemini.ts";

// ---------------------------------------------------------------------------
// Provider name tests
// ---------------------------------------------------------------------------

Deno.test("GoogleChatProvider - name is 'google'", () => {
  const provider = new GoogleChatProvider("test-key", "gemini-pro");
  assertEquals(provider.name, "google");
});

// ---------------------------------------------------------------------------
// convertMessages tests
// ---------------------------------------------------------------------------

Deno.test("convertMessages - text message", () => {
  const messages = [Message.user("Hello")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "user");
  assertEquals(converted[0].parts, [{ text: "Hello" }]);
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

Deno.test("convertMessages - assistant maps to 'model'", () => {
  const messages = [Message.assistant("I'm an assistant")];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].role, "model");
});

Deno.test("convertMessages - empty list", () => {
  const converted = convertMessages([]);
  assertEquals(converted.length, 0);
});

Deno.test("convertMessages - image block", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: "base64data",
          },
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].parts.length, 1);
  const part = converted[0].parts[0];
  assertEquals("inline_data" in part, true);
});

Deno.test("convertMessages - tool use block", () => {
  const messages = [
    new Message({
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "tool-123",
          name: "test_tool",
          input: { arg: "value" },
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].parts.length, 1);
  const part = converted[0].parts[0];
  assertEquals("function_call" in part, true);
});

Deno.test("convertMessages - tool result block", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "tool-123",
          content: "Result content",
        },
      ],
    }),
  ];
  const converted = convertMessages(messages);
  assertEquals(converted.length, 1);
  assertEquals(converted[0].parts.length, 1);
  const part = converted[0].parts[0];
  assertEquals("function_response" in part, true);
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
// getSystemInstruction tests
// ---------------------------------------------------------------------------

Deno.test("getSystemInstruction - found", () => {
  const messages = [
    Message.system("You are helpful"),
    Message.user("Hello"),
  ];
  assertEquals(getSystemInstruction(messages), "You are helpful");
});

Deno.test("getSystemInstruction - not found", () => {
  const messages = [Message.user("Hello")];
  assertEquals(getSystemInstruction(messages), undefined);
});

// ---------------------------------------------------------------------------
// buildGeminiRequest tests
// ---------------------------------------------------------------------------

Deno.test("buildGeminiRequest - minimal (no extra options)", () => {
  const messages = [Message.user("Hello")];
  // Build options with all generation-related fields explicitly undefined
  const options = new ChatOptions();
  options.temperature = undefined;
  options.max_tokens = undefined;
  options.top_p = undefined;
  const req = buildGeminiRequest(messages, undefined, options);
  assertEquals(req.contents.length, 1);
  assertEquals(req.systemInstruction, undefined);
  assertEquals(req.generationConfig, undefined);
  assertEquals(req.tools, undefined);
});

Deno.test("buildGeminiRequest - with system from options", () => {
  const messages = [Message.user("Hello")];
  const options = new ChatOptions({ system: "Be helpful" });
  const req = buildGeminiRequest(messages, undefined, options);
  assertEquals(req.systemInstruction !== undefined, true);
});

Deno.test("buildGeminiRequest - with generation config", () => {
  const messages = [Message.user("Hello")];
  const options = new ChatOptions({ temperature: 0.5, max_tokens: 1024 });
  const req = buildGeminiRequest(messages, undefined, options);
  assertEquals(req.generationConfig !== undefined, true);
  assertEquals(req.generationConfig!.temperature, 0.5);
  assertEquals(req.generationConfig!.maxOutputTokens, 1024);
});

Deno.test("buildGeminiRequest - with tools", () => {
  const messages = [Message.user("Hello")];
  const tools = [
    { name: "test", description: "Test", input_schema: { type: "object" } },
  ];
  const options = new ChatOptions();
  const req = buildGeminiRequest(messages, tools, options);
  assertEquals(req.tools !== undefined, true);
  assertEquals(req.tools!.length, 1);
});

// ---------------------------------------------------------------------------
// convertCandidateContent tests
// ---------------------------------------------------------------------------

Deno.test("convertCandidateContent - single text", () => {
  const parts = [{ text: "Hello world" }];
  const content = convertCandidateContent(parts);
  assertEquals(content, "Hello world");
});

Deno.test("convertCandidateContent - multiple parts", () => {
  const parts = [{ text: "Part 1" }, { text: "Part 2" }];
  const content = convertCandidateContent(parts);
  assertEquals(typeof content !== "string", true);
  if (typeof content !== "string") {
    assertEquals(content.length, 2);
  }
});

Deno.test("convertCandidateContent - function call", () => {
  const parts = [
    { function_call: { name: "do_thing", args: { a: 1 } } },
  ];
  const content = convertCandidateContent(parts);
  assertEquals(typeof content !== "string", true);
  if (typeof content !== "string") {
    assertEquals(content.length, 1);
    assertEquals(content[0].type, "tool_use");
  }
});

// ---------------------------------------------------------------------------
// parseGeminiResponse tests
// ---------------------------------------------------------------------------

Deno.test("parseGeminiResponse - basic text", () => {
  const response = {
    candidates: [{
      content: { parts: [{ text: "Hello!" }] },
      finishReason: "STOP",
    }],
    usageMetadata: {
      promptTokenCount: 10,
      candidatesTokenCount: 5,
      totalTokenCount: 15,
    },
  };
  const chatResponse = parseGeminiResponse(response);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(chatResponse.message.text(), "Hello!");
  assertEquals(chatResponse.usage.prompt_tokens, 10);
  assertEquals(chatResponse.usage.completion_tokens, 5);
  assertEquals(chatResponse.usage.total_tokens, 15);
  assertEquals(chatResponse.finish_reason, "STOP");
});
