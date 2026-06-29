// deno-lint-ignore-file no-explicit-any
/**
 * Anthropic (Claude) chat provider implementation.
 * Uses fetch() to api.anthropic.com/v1/messages.
 * Handles both streaming (SSE) and non-streaming responses.
 * Equivalent to Rust's `anthropic/mod.rs` + `anthropic/chat.rs`.
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

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";

// ---------------------------------------------------------------------------
// Anthropic wire types
// ---------------------------------------------------------------------------

interface AnthropicMessage {
  role: string;
  content: AnthropicContentBlock[];
}

type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: any }
  | { type: "tool_result"; tool_use_id: string; content: string };

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: Record<string, any>;
}

interface AnthropicResponse {
  content: AnthropicContentBlock[];
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number };
}

interface AnthropicStreamEvent {
  type: string;
  delta?: { text?: string };
  usage?: { input_tokens: number; output_tokens: number };
}

// ---------------------------------------------------------------------------
// AnthropicChatProvider
// ---------------------------------------------------------------------------

/** High-level Anthropic chat provider implementing the Provider interface.
 * Equivalent to Rust's `AnthropicChatProvider`. */
export class AnthropicChatProvider implements Provider {
  readonly name: string;
  private readonly apiKey: string;
  private readonly model: string;

  constructor(apiKey: string, model: string, providerName?: string) {
    this.apiKey = apiKey;
    this.model = model;
    this.name = providerName ?? "anthropic";
  }

  /** Create a copy with a different provider name (for Bedrock/Vertex variants). */
  withProviderName(name: string): AnthropicChatProvider {
    return new AnthropicChatProvider(this.apiKey, this.model, name);
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

    const response = await fetch(ANTHROPIC_API_URL, {
      method: "POST",
      headers: {
        "x-api-key": this.apiKey,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Anthropic API error (${response.status}): ${errorText}`,
      );
    }

    const anthropicResponse: AnthropicResponse = await response.json();
    return parseAnthropicResponse(anthropicResponse);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const body = this.buildRequestBody(messages, tools, options, true);

    const response = await fetch(ANTHROPIC_API_URL, {
      method: "POST",
      headers: {
        "x-api-key": this.apiKey,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Anthropic API error (${response.status}): ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error("Anthropic streaming response has no body");
    }

    for await (const data of parseSSEStream(response.body)) {
      let event: AnthropicStreamEvent;
      try {
        event = JSON.parse(data);
      } catch {
        continue;
      }

      switch (event.type) {
        case "content_block_delta":
          if (event.delta?.text) {
            yield { type: "text", text: event.delta.text };
          }
          break;
        case "message_delta":
          if (event.usage) {
            yield {
              type: "usage",
              usage: {
                prompt_tokens: 0,
                completion_tokens: event.usage.output_tokens,
                total_tokens: event.usage.output_tokens,
              },
            };
          }
          break;
        case "message_stop":
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
    const anthropicMessages = convertMessages(messages);
    const system = options.system ?? getSystemMessage(messages);

    const body: Record<string, any> = {
      model: this.model,
      messages: anthropicMessages,
      max_tokens: options.max_tokens ?? 4096,
      stream,
    };

    if (system) body.system = system;
    if (options.temperature !== undefined) {
      body.temperature = options.temperature;
    }
    if (options.top_p !== undefined) body.top_p = options.top_p;

    if (tools && tools.length > 0) {
      body.tools = convertTools(tools);
    }

    return body;
  }
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Convert core Messages to Anthropic wire format. */
export function convertMessages(messages: Message[]): AnthropicMessage[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => {
      const role = m.role === "assistant" ? "assistant" : "user";
      let content: AnthropicContentBlock[];

      if (typeof m.content === "string") {
        content = [{ type: "text", text: m.content }];
      } else {
        content = m.content
          .map((block): AnthropicContentBlock | null => {
            switch (block.type) {
              case "text":
                return { type: "text", text: block.text };
              case "tool_use":
                return {
                  type: "tool_use",
                  id: block.id,
                  name: block.name,
                  input: block.input,
                };
              case "tool_result":
                return {
                  type: "tool_result",
                  tool_use_id: block.tool_use_id,
                  content: block.content,
                };
              default:
                return null;
            }
          })
          .filter((b): b is AnthropicContentBlock => b !== null);
      }

      return { role, content };
    });
}

/** Convert core Tools to Anthropic wire format. */
export function convertTools(tools: Tool[]): AnthropicTool[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema.properties ?? {},
  }));
}

/** Extract the first system message from the message list. */
export function getSystemMessage(messages: Message[]): string | undefined {
  const sys = messages.find((m) => m.role === "system");
  if (!sys) return undefined;
  return typeof sys.content === "string" ? sys.content : undefined;
}

/** Parse an AnthropicResponse into a core ChatResponse. */
export function parseAnthropicResponse(
  response: AnthropicResponse,
): ChatResponse {
  let content: MessageContent;

  if (response.content.length === 1 && response.content[0].type === "text") {
    content = response.content[0].text;
  } else {
    content = response.content
      .map((block): ContentBlock | null => {
        switch (block.type) {
          case "text":
            return { type: "text", text: block.text };
          case "tool_use":
            return {
              type: "tool_use",
              id: block.id,
              name: block.name,
              input: block.input,
            };
          default:
            return null;
        }
      })
      .filter((b): b is ContentBlock => b !== null);
  }

  const usage: Usage = {
    prompt_tokens: response.usage.input_tokens,
    completion_tokens: response.usage.output_tokens,
    total_tokens: response.usage.input_tokens + response.usage.output_tokens,
  };

  return {
    message: new Message({
      role: "assistant",
      content,
    }),
    usage,
    finish_reason: response.stop_reason,
  };
}
