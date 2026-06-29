import { assertEquals } from "@std/assert";
import { Message } from "@rullama/core";
import {
  convertMessages,
  convertTools,
  OllamaChatProvider,
  parseOllamaResponse,
} from "./ollama.ts";

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
