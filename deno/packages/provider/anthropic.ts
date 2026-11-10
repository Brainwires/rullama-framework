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

/**
 * High-level Anthropic chat provider implementing the Provider interface.
 * Equivalent to Rust's `AnthropicChatProvider`.
 *
 * - Each reply keeps its raw content blocks (thinking, redacted thinking,
 *   signatures) in `message.metadata.anthropicContent`
 *   ({@link ANTHROPIC_CONTENT_KEY}); streamed replies deliver them in a
 *   `metadata` chunk before `done`. Keep that metadata on the message in the
 *   history and the next request sends the reply back exactly as it came.
 * - `temperature`, `top_p` and `top_k` are sent only when the caller set
 *   them (current models refuse sampling parameters); `ChatOptions`'
 *   default temperature is not sent.
 * - `ChatOptions.signal` aborts the request, streaming included.
 * - A reply's `stop_details` (e.g. a refusal's category) land in
 *   `message.metadata.anthropicStopDetails` ({@link ANTHROPIC_STOP_DETAILS_KEY}).
 * - Extra headers and body fields ({@link AnthropicProviderOptions}, and per
 *   call `ChatOptions.extra_headers` / `extra_body`) are sent on `chat` and
 *   `streamChat` alike; the provider's own headers and fields win over them.
 */
export class AnthropicChatProvider implements Provider {
  readonly name: string;
  private readonly apiKey: AnthropicApiKey;
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly extraHeaders: Record<string, string>;
  private readonly extraBody: Record<string, unknown>;
  private readonly fetchFn?: typeof fetch;

  /**
   * Create a provider bound to one Claude model, either positionally or with
   * one {@link AnthropicProviderOptions} object.
   *
   * @param apiKey Anthropic API key, sent as the `x-api-key` header; or a
   *   function returning it, called on every request.
   * @param model Model id (e.g. `claude-opus-5-5`).
   * @param providerName Value reported as `name` (default: `"anthropic"`);
   *   lets one wire format serve several registry entries.
   * @param baseUrl Messages endpoint (default: `https://api.anthropic.com/v1/messages`);
   *   set it to route through a gateway or proxy.
   * @param options Extra headers / body fields and an injected `fetch`.
   */
  constructor(
    apiKey: AnthropicApiKey,
    model: string,
    providerName?: string,
    baseUrl?: string,
    options?: AnthropicRequestOptions,
  );
  /** Create a provider from one options object. */
  constructor(options: AnthropicProviderOptions);
  constructor(
    keyOrOptions: AnthropicApiKey | AnthropicProviderOptions,
    model?: string,
    providerName?: string,
    baseUrl?: string,
    options?: AnthropicRequestOptions,
  ) {
    const o: AnthropicProviderOptions = typeof keyOrOptions === "object"
      ? keyOrOptions
      : {
        ...options,
        apiKey: keyOrOptions,
        model: model ?? "",
        providerName,
        baseUrl,
      };
    this.apiKey = o.apiKey;
    this.model = o.model;
    this.name = o.providerName ?? "anthropic";
    this.baseUrl = o.baseUrl ?? ANTHROPIC_API_URL;
    this.extraHeaders = { ...o.extraHeaders };
    this.extraBody = { ...o.extraBody };
    this.fetchFn = o.fetch;
  }

  /** Create a copy with a different provider name (all other options kept). */
  withProviderName(name: string): AnthropicChatProvider {
    return new AnthropicChatProvider({
      apiKey: this.apiKey,
      model: this.model,
      providerName: name,
      baseUrl: this.baseUrl,
      extraHeaders: this.extraHeaders,
      extraBody: this.extraBody,
      fetch: this.fetchFn,
    });
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
      options,
    );
    return parseAnthropicResponse(await response.json());
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const response = await this.post(
      this.body(messages, tools, options, true),
      options,
    );
    if (!response.body) {
      throw new Error("Anthropic streaming response has no body");
    }
    const state: AnthropicStreamState = { promptTokens: 0 };
    for await (const data of parseSSEStream(response.body)) {
      let event: AnthropicStreamEvent & { error?: { message?: string } };
      try {
        event = JSON.parse(data);
      } catch {
        continue;
      }
      if (event.type === "error") {
        // e.g. `overloaded_error` mid-stream: the reply is incomplete.
        throw new Error(
          `Anthropic stream error: ${event.error?.message ?? data}`,
        );
      }
      yield* streamEventToChunks(event, state);
    }
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  /** Build the Messages API JSON body (system prompt, messages, tools,
   * sampling options) for this provider's model, with `stream` set as given.
   * Extra body fields (provider's, then the call's) go first; the built
   * fields win over them. */
  private body(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
    stream: boolean,
  ): Record<string, unknown> {
    return {
      ...this.extraBody,
      ...options.extra_body,
      ...buildAnthropicBody(messages, tools, options, {
        model: this.model,
        stream,
      }),
    };
  }

  /** POST a Messages request (aborted by `options.signal`) with the key
   * resolved now; throws with the response body on a non-2xx status. Extra
   * headers (provider's, then the call's) go first; the auth and version
   * headers win over them. */
  private async post(
    body: Record<string, unknown>,
    options: ChatOptions,
  ): Promise<Response> {
    const key = typeof this.apiKey === "function"
      ? await this.apiKey()
      : this.apiKey;
    return postJson(
      "Anthropic",
      this.baseUrl,
      {
        ...this.extraHeaders,
        ...options.extra_headers,
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
      },
      body,
      options.signal,
      this.fetchFn,
    );
  }
}

/** An API key, or a function returning it (called on every request). */
export type AnthropicApiKey = string | (() => string | Promise<string>);

/** Request customization shared by both constructor forms. */
export interface AnthropicRequestOptions {
  /**
   * Headers added to every request, `chat` and `streamChat` alike (e.g.
   * `{ "anthropic-beta": "server-side-fallback-2026-07-01" }`). Per-call
   * `ChatOptions.extra_headers` are merged over them; `x-api-key`,
   * `anthropic-version` and `Content-Type` always win.
   */
  extraHeaders?: Record<string, string>;
  /**
   * Top-level fields added to every request body (e.g.
   * `{ fallbacks: "default" }`). Per-call `ChatOptions.extra_body` is merged
   * over them; the fields the provider builds (`model`, `messages`,
   * `system`, `max_tokens`, sampling, `stop_sequences`, `tools`, `stream`)
   * always win.
   */
  extraBody?: Record<string, unknown>;
  /** The `fetch` to call, for `chat` and `streamChat` (default: the global one). */
  fetch?: typeof fetch;
}

/** Everything an {@link AnthropicChatProvider} is built from. */
export interface AnthropicProviderOptions extends AnthropicRequestOptions {
  /** API key, or a function returning it (called on every request). */
  apiKey: AnthropicApiKey;
  /** Model id (e.g. `claude-opus-5-5`). */
  model: string;
  /** Value reported as `name` (default: `"anthropic"`). */
  providerName?: string;
  /** Messages endpoint (default: `https://api.anthropic.com/v1/messages`). */
  baseUrl?: string;
}

// Conversion helpers live in anthropic_format.ts (shared with Bedrock); re-exported
// so `import { convertMessages } from "./anthropic.ts"` keeps working.
export {
  ANTHROPIC_CONTENT_KEY,
  ANTHROPIC_STOP_DETAILS_KEY,
  type AnthropicContentBlock,
  type AnthropicMessage,
  type AnthropicRawBlock,
  type AnthropicResponse,
  type AnthropicStopDetails,
  type AnthropicStreamEvent,
  type AnthropicTool,
  convertMessages,
  convertTools,
  getSystemMessage,
  parseAnthropicResponse,
} from "./anthropic_format.ts";
