// deno-lint-ignore-file no-explicit-any
/**
 * OpenAI (and OpenAI-compatible) chat provider implementation.
 * Uses fetch() to OpenAI-compatible chat completions APIs.
 * Covers OpenAI, Groq, Together, Fireworks, Anyscale via base_url config.
 * Equivalent to Rust's `openai_chat/mod.rs` + `openai_chat/chat.rs`.
 */

import {
  type ChatOptions,
  type ChatResponse,
  type ContentBlock,
  Message,
  type MessageContent,
  type Provider,
  type StreamChunk,
  type Tool,
  type Usage,
} from "@rullama/core";
import { parseSSEStream } from "./sse.ts";

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";

// ---------------------------------------------------------------------------
// OpenAI wire types
// ---------------------------------------------------------------------------

interface OpenAIMessage {
  role: string;
  content: string | OpenAIContentPart[];
  name?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

interface OpenAIContentPart {
  type: string;
  text?: string;
  image_url?: { url: string };
}

interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
}

interface OpenAIToolCall {
  id?: string;
  type: string;
  function: { name?: string; arguments?: string };
}

interface OpenAIResponse {
  choices: Array<{
    message: {
      content: string | OpenAIContentPart[] | null;
      tool_calls?: OpenAIToolCall[];
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIStreamChunk {
  choices: Array<{
    delta?: {
      content?: string;
      tool_calls?: OpenAIToolCall[];
    };
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ---------------------------------------------------------------------------
// OpenAiChatProvider
// ---------------------------------------------------------------------------

/** High-level OpenAI chat provider implementing the Provider interface.
 * Works with any OpenAI-compatible API (Groq, Together, Fireworks, Anyscale).
 * Equivalent to Rust's `OpenAiChatProvider`. */
export class OpenAiChatProvider implements Provider {
  readonly name: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(
    apiKey: string,
    model: string,
    baseUrl?: string,
    providerName?: string,
  ) {
    this.apiKey = apiKey;
    this.model = model;
    this.baseUrl = baseUrl ?? OPENAI_API_URL;
    this.name = providerName ?? "openai";
  }

  /** Create a copy with a different provider name. */
  withProviderName(name: string): OpenAiChatProvider {
    return new OpenAiChatProvider(
      this.apiKey,
      this.model,
      this.baseUrl,
      name,
    );
  }

  /** Check if a model is an O1/O3 model (no streaming, no system messages). */
  static isO1Model(model: string): boolean {
    return model.startsWith("o1-") || model.startsWith("o3-");
  }

  // -----------------------------------------------------------------------
  // Provider interface
  // -----------------------------------------------------------------------

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    const body = this.buildRequestBody(messages, tools, options, false);

    const response = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenAI API error (${response.status}): ${errorText}`,
      );
    }

    const openaiResponse: OpenAIResponse = await response.json();
    return parseOpenAIResponse(openaiResponse);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    // O1 models don't support streaming - fall back to non-streaming
    if (OpenAiChatProvider.isO1Model(this.model)) {
      const response = await this.chat(messages, tools, options);
      const text = response.message.text();
      if (text) {
        yield { type: "text", text };
      }
      yield { type: "usage", usage: response.usage };
      yield { type: "done" };
      return;
    }

    const body = this.buildRequestBody(messages, tools, options, true);

    const response = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenAI API error (${response.status}): ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error("OpenAI streaming response has no body");
    }

    for await (const data of parseSSEStream(response.body)) {
      let chunk: OpenAIStreamChunk;
      try {
        chunk = JSON.parse(data);
      } catch {
        continue;
      }

      for (const streamChunk of convertStreamChunk(chunk)) {
        yield streamChunk;
      }
    }

    yield { type: "done" };
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  private buildRequestBody(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
    stream: boolean,
  ): Record<string, any> {
    const openaiMessages = convertMessages(messages);

    const body: Record<string, any> = {
      model: this.model,
      messages: openaiMessages,
    };

    if (stream) body.stream = true;

    if (!OpenAiChatProvider.isO1Model(this.model)) {
      if (options.max_tokens !== undefined) {
        body.max_tokens = options.max_tokens;
      }
      if (options.temperature !== undefined) {
        body.temperature = options.temperature;
      }
      if (options.top_p !== undefined) body.top_p = options.top_p;
      if (options.stop) body.stop = options.stop;
    }

    if (tools && tools.length > 0) {
      body.tools = convertTools(tools);
    }

    return body;
  }
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Convert core Messages to OpenAI wire format. */
export function convertMessages(messages: Message[]): OpenAIMessage[] {
  return messages.map((m) => {
    const role = m.role;
    let content: string | OpenAIContentPart[];

    if (typeof m.content === "string") {
      content = m.content;
    } else if (m.content.length === 1 && m.content[0].type === "text") {
      content = m.content[0].text;
    } else {
      content = m.content
        .map((block): OpenAIContentPart | null => {
          switch (block.type) {
            case "text":
              return { type: "text", text: block.text };
            case "image":
              return {
                type: "image_url",
                image_url: {
                  url:
                    `data:${block.source.media_type};base64,${block.source.data}`,
                },
              };
            default:
              return null;
          }
        })
        .filter((b): b is OpenAIContentPart => b !== null);
    }

    const msg: OpenAIMessage = { role, content };
    if (m.name) msg.name = m.name;
    return msg;
  });
}

/** Convert core Tools to OpenAI wire format. */
export function convertTools(tools: Tool[]): OpenAITool[] {
  return tools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema.properties ?? {},
    },
  }));
}

/** Parse an OpenAI response into a core ChatResponse. */
export function parseOpenAIResponse(
  openaiResponse: OpenAIResponse,
): ChatResponse {
  const usage: Usage = {
    prompt_tokens: openaiResponse.usage.prompt_tokens,
    completion_tokens: openaiResponse.usage.completion_tokens,
    total_tokens: openaiResponse.usage.total_tokens,
  };

  const choice = openaiResponse.choices[0];
  if (!choice) {
    throw new Error("No choices in OpenAI response");
  }

  let content: MessageContent;
  const rawContent = choice.message.content;

  if (rawContent === null || rawContent === undefined) {
    content = "";
  } else if (typeof rawContent === "string") {
    content = rawContent;
  } else {
    content = rawContent
      .map((part): ContentBlock | null => {
        if (part.type === "text" && part.text !== undefined) {
          return { type: "text", text: part.text };
        }
        return null;
      })
      .filter((b): b is ContentBlock => b !== null);
  }

  return {
    message: new Message({ role: "assistant", content }),
    usage,
    finish_reason: choice.finish_reason,
  };
}

/** Convert a raw OpenAI stream chunk into zero or more StreamChunk values. */
export function convertStreamChunk(chunk: OpenAIStreamChunk): StreamChunk[] {
  const out: StreamChunk[] = [];

  for (const choice of chunk.choices) {
    if (choice.delta) {
      if (choice.delta.content) {
        out.push({ type: "text", text: choice.delta.content });
      }
      if (choice.delta.tool_calls) {
        for (const tc of choice.delta.tool_calls) {
          out.push({
            type: "tool_use",
            id: tc.id ?? "",
            name: tc.function.name ?? "",
          });
        }
      }
    }
  }

  if (chunk.usage) {
    out.push({
      type: "usage",
      usage: {
        prompt_tokens: chunk.usage.prompt_tokens,
        completion_tokens: chunk.usage.completion_tokens,
        total_tokens: chunk.usage.total_tokens,
      },
    });
  }

  return out;
}
