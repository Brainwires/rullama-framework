// deno-lint-ignore-file no-explicit-any
/**
 * Google Gemini chat provider implementation.
 * Uses fetch() to the Gemini generateContent API.
 * Equivalent to Rust's `gemini/mod.rs` + `gemini/chat.rs`.
 */

import { postJson } from "./http.ts";
import { mapBlocks, textBlock, toolUseBlock } from "./content.ts";
import {
  type ChatOptions,
  type ChatResponse,
  Message,
  type MessageContent,
  type Provider,
  type StreamChunk,
  type Tool,
  toolInputJsonSchema,
  type Usage,
} from "@rullama/core";
import { parseNDJSONStream } from "./sse.ts";

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta";

// ---------------------------------------------------------------------------
// Gemini wire types
// ---------------------------------------------------------------------------

interface GeminiRequest {
  contents: GeminiMessage[];
  systemInstruction?: { parts: GeminiPart[] };
  generationConfig?: {
    temperature?: number;
    maxOutputTokens?: number;
    topP?: number;
    stopSequences?: string[];
  };
  tools?: Array<{
    function_declarations: GeminiFunctionDeclaration[];
  }>;
}

interface GeminiMessage {
  role: string;
  parts: GeminiPart[];
}

type GeminiPart =
  | { text: string }
  | { inline_data: { mime_type: string; data: string } }
  | { function_call: { name: string; args: any } }
  | { function_response: { name: string; response: any } };

interface GeminiFunctionDeclaration {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

interface GeminiResponse {
  candidates: Array<{
    content: { parts: GeminiPart[] };
    finishReason: string;
  }>;
  usageMetadata?: {
    promptTokenCount: number;
    candidatesTokenCount: number;
    totalTokenCount: number;
  };
}

interface GeminiStreamChunk {
  candidates: Array<{
    content: { parts: GeminiPart[] };
    finishReason: string;
  }>;
  usageMetadata?: {
    promptTokenCount: number;
    candidatesTokenCount: number;
    totalTokenCount: number;
  };
}

// ---------------------------------------------------------------------------
// GoogleChatProvider
// ---------------------------------------------------------------------------

/** High-level Google Gemini chat provider implementing the Provider interface.
 * Equivalent to Rust's `GoogleChatProvider`. */
export class GoogleChatProvider implements Provider {
  readonly name = "google";
  private readonly apiKey: string;
  private readonly model: string;

  constructor(apiKey: string, model: string) {
    this.apiKey = apiKey;
    this.model = model;
  }

  /** POST a Gemini request; throws with the response body on a non-2xx status. */
  private post(url: string, request: GeminiRequest): Promise<Response> {
    return postJson(
      "Google Gemini",
      url,
      { "x-goog-api-key": this.apiKey },
      request,
    );
  }

  // -----------------------------------------------------------------------
  // Provider interface
  // -----------------------------------------------------------------------

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    const request = buildGeminiRequest(messages, tools, options);
    const url = `${GEMINI_API_BASE}/models/${this.model}:generateContent`;

    const response = await this.post(url, request);

    const geminiResponse: GeminiResponse = await response.json();
    return parseGeminiResponse(geminiResponse);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const request = buildGeminiRequest(messages, tools, options);
    const url = `${GEMINI_API_BASE}/models/${this.model}:streamGenerateContent`;

    const response = await this.post(url, request);

    if (!response.body) {
      throw new Error("Google Gemini streaming response has no body");
    }

    for await (const line of parseNDJSONStream(response.body)) {
      let chunk: GeminiStreamChunk;
      try {
        chunk = JSON.parse(line);
      } catch {
        continue;
      }

      const candidate = chunk.candidates?.[0];
      if (candidate) {
        for (const part of candidate.content.parts) {
          if ("text" in part) {
            yield { type: "text", text: part.text };
          } else if ("function_call" in part) {
            yield {
              type: "tool_use",
              id: crypto.randomUUID(),
              name: part.function_call.name,
            };
          }
        }
      }

      if (chunk.usageMetadata) {
        yield {
          type: "usage",
          usage: {
            prompt_tokens: chunk.usageMetadata.promptTokenCount,
            completion_tokens: chunk.usageMetadata.candidatesTokenCount,
            total_tokens: chunk.usageMetadata.totalTokenCount,
          },
        };
      }
    }

    yield { type: "done" };
  }
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Convert core Messages to Gemini wire format. */
export function convertMessages(messages: Message[]): GeminiMessage[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => {
      const role = m.role === "assistant" ? "model" : "user";
      let parts: GeminiPart[];

      if (typeof m.content === "string") {
        parts = [{ text: m.content }];
      } else {
        parts = m.content
          .map((block): GeminiPart | null => {
            switch (block.type) {
              case "text":
                return { text: block.text };
              case "image":
                return {
                  inline_data: {
                    mime_type: block.source.media_type,
                    data: block.source.data,
                  },
                };
              case "tool_use":
                return {
                  function_call: {
                    name: block.name,
                    args: block.input,
                  },
                };
              case "tool_result":
                return {
                  function_response: {
                    name: block.tool_use_id,
                    response: { result: block.content },
                  },
                };
              default:
                return null;
            }
          })
          .filter((p): p is GeminiPart => p !== null);
      }

      return { role, parts };
    });
}

/** Convert core Tools to Gemini function declarations. */
export function convertTools(tools: Tool[]): GeminiFunctionDeclaration[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    parameters: toolInputJsonSchema(t.input_schema),
  }));
}

/** Extract system instruction from message list. */
export function getSystemInstruction(
  messages: Message[],
): string | undefined {
  const sys = messages.find((m) => m.role === "system");
  if (!sys) return undefined;
  return typeof sys.content === "string" ? sys.content : undefined;
}

/** Build a Gemini request from core types. */
export function buildGeminiRequest(
  messages: Message[],
  tools: Tool[] | undefined,
  options: ChatOptions,
): GeminiRequest {
  const contents = convertMessages(messages);
  const systemText = options.system ?? getSystemInstruction(messages);

  const request: GeminiRequest = { contents };

  if (systemText) {
    request.systemInstruction = { parts: [{ text: systemText }] };
  }

  const hasConfig = options.temperature !== undefined ||
    options.max_tokens !== undefined ||
    options.top_p !== undefined || options.stop !== undefined;

  if (hasConfig) {
    request.generationConfig = {};
    if (options.temperature !== undefined) {
      request.generationConfig.temperature = options.temperature;
    }
    if (options.max_tokens !== undefined) {
      request.generationConfig.maxOutputTokens = options.max_tokens;
    }
    if (options.top_p !== undefined) {
      request.generationConfig.topP = options.top_p;
    }
    if (options.stop !== undefined) {
      request.generationConfig.stopSequences = options.stop;
    }
  }

  if (tools && tools.length > 0) {
    request.tools = [{ function_declarations: convertTools(tools) }];
  }

  return request;
}

/** Convert Gemini candidate content parts to core MessageContent. */
export function convertCandidateContent(
  parts: GeminiPart[],
): MessageContent {
  return mapBlocks(
    parts,
    (part) =>
      "text" in part
        ? textBlock(part.text)
        : "function_call" in part
        ? toolUseBlock(
          crypto.randomUUID(),
          part.function_call.name,
          part.function_call.args,
        )
        : null,
  );
}

/** Parse a GeminiResponse into a core ChatResponse. */
export function parseGeminiResponse(
  geminiResponse: GeminiResponse,
): ChatResponse {
  const candidate = geminiResponse.candidates[0];
  if (!candidate) {
    throw new Error("No candidates in Gemini response");
  }

  const content = convertCandidateContent(candidate.content.parts);

  const usage: Usage = geminiResponse.usageMetadata
    ? {
      prompt_tokens: geminiResponse.usageMetadata.promptTokenCount,
      completion_tokens: geminiResponse.usageMetadata.candidatesTokenCount,
      total_tokens: geminiResponse.usageMetadata.totalTokenCount,
    }
    : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

  return {
    message: new Message({ role: "assistant", content }),
    usage,
    finish_reason: candidate.finishReason,
  };
}
