/**
 * Anthropic (Claude) chat provider implementation.
 * Uses fetch() to api.anthropic.com/v1/messages.
 * Handles both streaming (SSE) and non-streaming responses.
 * Equivalent to Rust's `anthropic/mod.rs` + `anthropic/chat.rs`.
 */

import { postJson } from "./http.ts";
import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@rullama/core";
import {
  type AnthropicStreamEvent,
  type AnthropicStreamState,
  buildAnthropicBody,
  parseAnthropicResponse,
  streamEventToChunks,
} from "./anthropic_format.ts";
import { parseSSEStream } from "./sse.ts";

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";

// ---------------------------------------------------------------------------
// AnthropicChatProvider
// ---------------------------------------------------------------------------

/** High-level Anthropic chat provider implementing the Provider interface.
 * Equivalent to Rust's `AnthropicChatProvider`. */
export class AnthropicChatProvider implements Provider {
  readonly name: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;

  /**
   * Create a provider bound to one Claude model.
   *
   * @param apiKey Anthropic API key, sent as the `x-api-key` header.
   * @param model Model id (e.g. `claude-sonnet-4-20250514`).
   * @param providerName Value reported as `name` (default: `"anthropic"`);
   *   lets one wire format serve several registry entries.
   * @param baseUrl Messages endpoint (default: `https://api.anthropic.com/v1/messages`);
   *   set it to route through a gateway or proxy.
   */
  constructor(
    apiKey: string,
    model: string,
    providerName?: string,
    baseUrl?: string,
  ) {
    this.apiKey = apiKey;
    this.model = model;
    this.name = providerName ?? "anthropic";
    this.baseUrl = baseUrl ?? ANTHROPIC_API_URL;
  }

  /** Create a copy with a different provider name. */
  withProviderName(name: string): AnthropicChatProvider {
    return new AnthropicChatProvider(
      this.apiKey,
      this.model,
      name,
      this.baseUrl,
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
    const response = await this.post(
      this.body(messages, tools, options, false),
    );
    return parseAnthropicResponse(await response.json());
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const response = await this.post(this.body(messages, tools, options, true));
    if (!response.body) {
      throw new Error("Anthropic streaming response has no body");
    }
    const state: AnthropicStreamState = { promptTokens: 0 };
    for await (const data of parseSSEStream(response.body)) {
      let event: AnthropicStreamEvent;
      try {
        event = JSON.parse(data);
      } catch {
        continue;
      }
      yield* streamEventToChunks(event, state);
    }
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  /** Build the Messages API JSON body (system prompt, messages, tools,
   * sampling options) for this provider's model, with `stream` set as given. */
  private body(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
    stream: boolean,
  ): Record<string, unknown> {
    return buildAnthropicBody(messages, tools, options, {
      model: this.model,
      stream,
    });
  }

  /** POST a Messages request; throws with the response body on a non-2xx status. */
  private post(body: Record<string, unknown>): Promise<Response> {
    return postJson("Anthropic", this.baseUrl, {
      "x-api-key": this.apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
    }, body);
  }
}

// Conversion helpers live in anthropic_format.ts (shared with Bedrock); re-exported
// so `import { convertMessages } from "./anthropic.ts"` keeps working.
export {
  type AnthropicContentBlock,
  type AnthropicMessage,
  type AnthropicResponse,
  type AnthropicStreamEvent,
  type AnthropicTool,
  convertMessages,
  convertTools,
  getSystemMessage,
  parseAnthropicResponse,
} from "./anthropic_format.ts";
