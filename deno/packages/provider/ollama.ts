// deno-lint-ignore-file no-explicit-any
/**
 * Ollama local model chat provider implementation.
 * Uses fetch() to the local Ollama API.
 * Equivalent to Rust's `ollama/mod.rs` + `ollama/chat.rs`.
 */

import {
  type ChatOptions,
  type ChatResponse,
  Message,
  type MessageContent,
  type Provider,
  type StreamChunk,
  type Tool,
  type Usage,
} from "@rullama/core";
import { parseNDJSONStream } from "./sse.ts";

const DEFAULT_OLLAMA_URL = "http://localhost:11434";

// ---------------------------------------------------------------------------
// Ollama wire types
// ---------------------------------------------------------------------------

interface OllamaMessage {
  role: string;
  content: string;
  images?: string[];
}

interface OllamaTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, any>;
      required: string[];
    };
  };
}

interface OllamaResponse {
  message: { content: string };
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
}

interface OllamaStreamChunk {
  message?: { content: string };
  done: boolean;
  prompt_eval_count?: number;
  eval_count?: number;
}

// ---------------------------------------------------------------------------
// OllamaChatProvider
// ---------------------------------------------------------------------------

/** High-level Ollama chat provider implementing the Provider interface.
 * Equivalent to Rust's `OllamaChatProvider`. */
export class OllamaChatProvider implements Provider {
  readonly name = "ollama";
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(model: string, baseUrl?: string) {
    this.model = model;
    this.baseUrl = baseUrl ?? DEFAULT_OLLAMA_URL;
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
    const url = `${this.baseUrl}/api/chat`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Ollama API error (${response.status}): ${errorText}`,
      );
    }

    const ollamaResponse: OllamaResponse = await response.json();
    return parseOllamaResponse(ollamaResponse);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const body = this.buildRequestBody(messages, tools, options, true);
    const url = `${this.baseUrl}/api/chat`;

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Ollama API error (${response.status}): ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error("Ollama streaming response has no body");
    }

    for await (const line of parseNDJSONStream(response.body)) {
      let chunk: OllamaStreamChunk;
      try {
        chunk = JSON.parse(line);
      } catch {
        continue;
      }

      if (chunk.message && chunk.message.content.length > 0) {
        yield { type: "text", text: chunk.message.content };
      }

      if (chunk.done) {
        if (
          chunk.prompt_eval_count !== undefined &&
          chunk.eval_count !== undefined
        ) {
          const promptTokens = chunk.prompt_eval_count;
          const completionTokens = chunk.eval_count;
          yield {
            type: "usage",
            usage: {
              prompt_tokens: promptTokens,
              completion_tokens: completionTokens,
              total_tokens: promptTokens + completionTokens,
            },
          };
        }
        yield { type: "done" };
        break;
      }
    }
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
    const ollamaMessages = convertMessages(messages);

    const body: Record<string, any> = {
      model: this.model,
      messages: ollamaMessages,
      stream,
    };

    const opts: Record<string, any> = {};
    if (options.temperature !== undefined) {
      opts.temperature = options.temperature;
    }
    if (options.top_p !== undefined) opts.top_p = options.top_p;
    if (Object.keys(opts).length > 0) body.options = opts;

    if (tools && tools.length > 0) {
      body.tools = convertTools(tools);
    }

    return body;
  }
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Convert core Messages to Ollama wire format. */
export function convertMessages(messages: Message[]): OllamaMessage[] {
  return messages.map((m) => {
    const role = m.role;
    let content: string;
    let images: string[] | undefined;

    if (typeof m.content === "string") {
      content = m.content;
    } else {
      const textParts: string[] = [];
      const imageData: string[] = [];

      for (const block of m.content) {
        switch (block.type) {
          case "text":
            textParts.push(block.text);
            break;
          case "image":
            imageData.push(block.source.data);
            break;
        }
      }

      content = textParts.join("\n");
      if (imageData.length > 0) images = imageData;
    }

    const msg: OllamaMessage = { role, content };
    if (images) msg.images = images;
    return msg;
  });
}

/** Convert core Tools to Ollama wire format. */
export function convertTools(tools: Tool[]): OllamaTool[] {
  return tools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: {
        type: "object" as const,
        properties: t.input_schema.properties ?? {},
        required: t.input_schema.required ?? [],
      },
    },
  }));
}

/** Parse an Ollama response into a core ChatResponse. */
export function parseOllamaResponse(
  response: OllamaResponse,
): ChatResponse {
  const content: MessageContent = response.message.content;
  const promptTokens = response.prompt_eval_count ?? 0;
  const completionTokens = response.eval_count ?? 0;

  const usage: Usage = {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };

  return {
    message: new Message({ role: "assistant", content }),
    usage,
    finish_reason: response.done_reason ?? "stop",
  };
}
