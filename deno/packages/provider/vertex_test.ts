import { assertEquals, assertThrows } from "@std/assert";
import { Message } from "@rullama/core";
import {
  convertMessages,
  convertTools,
  parseVertexResponse,
  vertexGenerateContentUrl,
  vertexStreamUrl,
} from "./vertex.ts";

// ---------------------------------------------------------------------------
// URL building tests
// ---------------------------------------------------------------------------

Deno.test("vertexGenerateContentUrl - correct format", () => {
  const url = vertexGenerateContentUrl(
    "us-central1",
    "my-project",
    "gemini-2.0-flash",
  );
  assertEquals(
    url,
    "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent",
  );
});

Deno.test("vertexStreamUrl - correct format", () => {
  const url = vertexStreamUrl("europe-west1", "proj-123", "gemini-pro");
  assertEquals(
    url,
    "https://europe-west1-aiplatform.googleapis.com/v1/projects/proj-123/locations/europe-west1/publishers/google/models/gemini-pro:streamGenerateContent?alt=sse",
  );
});

// ---------------------------------------------------------------------------
// convertMessages tests
// ---------------------------------------------------------------------------

Deno.test("convertMessages - text message", () => {
  const messages = [Message.user("Hello")];
  const [contents, system] = convertMessages(messages);
  assertEquals(contents.length, 1);
  assertEquals(contents[0].role, "user");
  assertEquals(contents[0].parts.length, 1);
  assertEquals(contents[0].parts[0].text, "Hello");
  assertEquals(system, undefined);
});

Deno.test("convertMessages - system message extracted", () => {
  const messages = [
    Message.system("You are helpful"),
    Message.user("Hello"),
  ];
  const [contents, system] = convertMessages(messages);
  assertEquals(contents.length, 1);
  assertEquals(system, "You are helpful");
});

Deno.test("convertMessages - assistant maps to model role", () => {
  const messages = [Message.assistant("I can help")];
  const [contents, _system] = convertMessages(messages);
  assertEquals(contents.length, 1);
  assertEquals(contents[0].role, "model");
});

Deno.test("convertMessages - tool use blocks become functionCall", () => {
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
  const [contents, _system] = convertMessages(messages);
  assertEquals(contents[0].parts.length, 2);
  assertEquals(contents[0].parts[0].text, "Let me search");
  assertEquals(contents[0].parts[1].functionCall?.name, "search");
  assertEquals(contents[0].parts[1].functionCall?.args.q, "test");
});

Deno.test("convertMessages - tool result blocks become functionResponse", () => {
  const messages = [
    new Message({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "search", content: "42 results" },
      ],
    }),
  ];
  const [contents, _system] = convertMessages(messages);
  assertEquals(contents[0].parts.length, 1);
  assertEquals(contents[0].parts[0].functionResponse?.name, "search");
  assertEquals(
    contents[0].parts[0].functionResponse?.response.result,
    "42 results",
  );
});

Deno.test("convertMessages - empty list", () => {
  const [contents, system] = convertMessages([]);
  assertEquals(contents.length, 0);
  assertEquals(system, undefined);
});

Deno.test("convertMessages - preserves conversation order", () => {
  const messages = [
    Message.user("Question"),
    Message.assistant("Answer"),
    Message.user("Follow-up"),
  ];
  const [contents, _system] = convertMessages(messages);
  assertEquals(contents.length, 3);
  assertEquals(contents[0].role, "user");
  assertEquals(contents[1].role, "model");
  assertEquals(contents[2].role, "user");
});

// ---------------------------------------------------------------------------
// convertTools tests
// ---------------------------------------------------------------------------

Deno.test("convertTools - basic tool", () => {
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
  const converted = convertTools(tools);
  assertEquals(converted.functionDeclarations.length, 1);
  assertEquals(converted.functionDeclarations[0].name, "search");
  assertEquals(converted.functionDeclarations[0].description, "Search the web");
  assertEquals(converted.functionDeclarations[0].parameters.type, "object");
});

Deno.test("convertTools - tool without properties", () => {
  const tools = [
    {
      name: "ping",
      description: "Ping",
      input_schema: { type: "object" },
    },
  ];
  const converted = convertTools(tools);
  assertEquals(converted.functionDeclarations[0].parameters.type, "object");
});

Deno.test("convertTools - multiple tools", () => {
  const tools = [
    {
      name: "tool1",
      description: "First",
      input_schema: { type: "object", properties: {} },
    },
    {
      name: "tool2",
      description: "Second",
      input_schema: { type: "object", properties: {} },
    },
  ];
  const converted = convertTools(tools);
  assertEquals(converted.functionDeclarations.length, 2);
});

// ---------------------------------------------------------------------------
// parseVertexResponse tests
// ---------------------------------------------------------------------------

Deno.test("parseVertexResponse - text response", () => {
  const resp = {
    candidates: [{
      content: {
        role: "model",
        parts: [{ text: "Hello!" }],
      },
      finishReason: "STOP",
    }],
    usageMetadata: {
      promptTokenCount: 10,
      candidatesTokenCount: 5,
      totalTokenCount: 15,
    },
  };
  const chatResponse = parseVertexResponse(resp);
  assertEquals(chatResponse.message.role, "assistant");
  assertEquals(chatResponse.message.text(), "Hello!");
  assertEquals(chatResponse.usage.prompt_tokens, 10);
  assertEquals(chatResponse.usage.completion_tokens, 5);
  assertEquals(chatResponse.usage.total_tokens, 15);
  assertEquals(chatResponse.finish_reason, "STOP");
});

Deno.test("parseVertexResponse - function call response", () => {
  const resp = {
    candidates: [{
      content: {
        role: "model",
        parts: [
          { text: "Let me search" },
          { functionCall: { name: "search", args: { q: "test" } } },
        ],
      },
      finishReason: "STOP",
    }],
  };
  const chatResponse = parseVertexResponse(resp);
  if (typeof chatResponse.message.content !== "string") {
    assertEquals(chatResponse.message.content.length, 2);
    assertEquals(chatResponse.message.content[0].type, "text");
    assertEquals(chatResponse.message.content[1].type, "tool_use");
    if (chatResponse.message.content[1].type === "tool_use") {
      assertEquals(chatResponse.message.content[1].name, "search");
    }
  }
});

Deno.test("parseVertexResponse - no candidates throws", () => {
  assertThrows(
    () => parseVertexResponse({ candidates: [] }),
    Error,
    "No candidates",
  );
});

Deno.test("parseVertexResponse - no usage metadata", () => {
  const resp = {
    candidates: [{
      content: { role: "model", parts: [{ text: "Hi" }] },
    }],
  };
  const chatResponse = parseVertexResponse(resp);
  assertEquals(chatResponse.usage.prompt_tokens, 0);
  assertEquals(chatResponse.usage.completion_tokens, 0);
  assertEquals(chatResponse.usage.total_tokens, 0);
});

Deno.test("parseVertexResponse - empty parts", () => {
  const resp = {
    candidates: [{
      content: { role: "model", parts: [] },
    }],
  };
  const chatResponse = parseVertexResponse(resp);
  assertEquals(chatResponse.message.text(), "");
});
