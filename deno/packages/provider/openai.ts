// deno-lint-ignore-file no-explicit-any
/**
 * OpenAI (and OpenAI-compatible) chat provider implementation.
 * Uses fetch() to OpenAI-compatible chat completions APIs.
 * Covers OpenAI, Groq, Together, Fireworks, Anyscale via base_url config.
 * Equivalent to Rust's `openai_chat/mod.rs` + `openai_chat/chat.rs`.
 */

import { postJson } from "./http.ts";
import { collapseBlocks } from "./content.ts";
import {
  type ChatOptions,
  type ChatResponse,
  type ContentBlock,
  Message,
  type MessageContent,
  type Provider,
  type StreamChunk,
  type Tool,
  toolInputJsonSchema,
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
  /** Position within a streamed tool-call list (streaming only). */
  index?: number;
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

  /**
   * Create a provider for an OpenAI-compatible Chat Completions endpoint.
   *
   * @param apiKey Bearer token for the `Authorization` header.
   * @param model Model id (e.g. `gpt-5-mini`, `llama-3.3-70b-versatile`).
   * @param baseUrl Full chat-completions URL (default:
   *   `https://api.openai.com/v1/chat/completions`); point it at Groq,
   *   Together, Fireworks, Anyscale or any compatible server.
   * @param providerName Value reported as `name` (default: `"openai"`).
   */
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

  /**
   * Reasoning models (`o1`, `o3`, `o4`, `gpt-5` families) take
   * `max_completion_tokens` and reject `temperature` / `top_p`.
   */
  static isReasoningModel(model: string): boolean {
    return /^(o[1-9](-|$)|gpt-5)/.test(model);
  }

  /** @deprecated Use {@link OpenAiChatProvider.isReasoningModel}. */
  static isO1Model(model: string): boolean {
    return OpenAiChatProvider.isReasoningModel(model);
  }

  /** POST a Chat Completions request; throws with the body on a non-2xx status. */
  private post(body: Record<string, any>): Promise<Response> {
    return postJson("OpenAI", this.baseUrl, {
      Authorization: `Bearer ${this.apiKey}`,
    }, body);
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

    const response = await this.post(body);

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

    const response = await this.post(body);

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

  /** Build the Chat Completions body: converted messages, sampling options
   * (`max_completion_tokens` only for reasoning models), function tools, and
   * `stream: true` when streaming. */
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

    applySamplingOptions(
      body,
      options,
      OpenAiChatProvider.isReasoningModel(this.model),
    );

    if (tools && tools.length > 0) {
      body.tools = convertTools(tools);
    }

    return body;
  }
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Reasoning models take `max_completion_tokens` and reject sampling knobs. */
function applySamplingOptions(
  body: Record<string, any>,
  options: ChatOptions,
  reasoning: boolean,
): void {
  if (reasoning) {
    if (options.max_tokens !== undefined) {
      body.max_completion_tokens = options.max_tokens;
    }
  } else {
    if (options.max_tokens !== undefined) body.max_tokens = options.max_tokens;
    if (options.temperature !== undefined) {
      body.temperature = options.temperature;
    }
    if (options.top_p !== undefined) body.top_p = options.top_p;
  }
  if (options.stop) body.stop = options.stop;
}

/** Text of a block list, or the whole string. */
function textOf(content: MessageContent): string {
  if (typeof content === "string") return content;
  return content
    .filter((b): b is Extract<ContentBlock, { type: "text" }> =>
      b.type === "text"
    )
    .map((b) => b.text)
    .join("");
}

/** Content parts (text + images) of a message; `null` when there are none. */
function partsOf(content: MessageContent): string | OpenAIContentPart[] | null {
  if (typeof content === "string") return content;
  const parts = content
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
  if (parts.length === 0) return null;
  if (parts.length === 1 && parts[0].type === "text") return parts[0].text!;
  return parts;
}

/**
 * Convert core Messages to OpenAI wire format.
 *
 * An assistant message's `tool_use` blocks become `tool_calls`; each
 * `tool_result` block becomes its own `role: "tool"` message keyed by
 * `tool_call_id` — the shape Chat Completions requires for a tool round-trip.
 */
function toolCallsOf(
  uses: Extract<ContentBlock, { type: "tool_use" }>[],
): OpenAIToolCall[] {
  return uses.map((t) => ({
    id: t.id,
    type: "function",
    function: { name: t.name, arguments: JSON.stringify(t.input ?? {}) },
  }));
}

/** The assistant/user/system message for `m`, or `null` if it is tool results only. */
function primaryMessage(
  m: Message,
  toolUses: Extract<ContentBlock, { type: "tool_use" }>[],
  hasToolResults: boolean,
): OpenAIMessage | null {
  const parts = partsOf(m.content);
  const plain = parts !== null || (toolUses.length === 0 && !hasToolResults);
  if (!plain && toolUses.length === 0) return null;
  const msg: OpenAIMessage = {
    role: plain ? m.role : "assistant",
    content: parts ?? "",
  };
  if (plain && m.name) msg.name = m.name;
  return withToolCalls(msg, toolUses);
}

/** Attach `tool_calls` when there are any. */
function withToolCalls(
  msg: OpenAIMessage,
  toolUses: Extract<ContentBlock, { type: "tool_use" }>[],
): OpenAIMessage {
  if (toolUses.length > 0) msg.tool_calls = toolCallsOf(toolUses);
  return msg;
}

/** Convert one core message to zero or more OpenAI messages. */
function convertMessage(m: Message): OpenAIMessage[] {
  const blocks = typeof m.content === "string" ? [] : m.content;
  const toolResults = blocks.filter(
    (b): b is Extract<ContentBlock, { type: "tool_result" }> =>
      b.type === "tool_result",
  );
  const toolUses = blocks.filter(
    (b): b is Extract<ContentBlock, { type: "tool_use" }> =>
      b.type === "tool_use",
  );
  const primary = primaryMessage(m, toolUses, toolResults.length > 0);
  const results = toolResults.map((r): OpenAIMessage => ({
    role: "tool",
    content: textOf(r.content as MessageContent),
    tool_call_id: r.tool_use_id,
  }));
  return primary ? [primary, ...results] : results;
}

export function convertMessages(messages: Message[]): OpenAIMessage[] {
  return messages.flatMap(convertMessage);
}

/** Convert core Tools to OpenAI wire format. */
export function convertTools(tools: Tool[]): OpenAITool[] {
  return tools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: toolInputJsonSchema(t.input_schema),
    },
  }));
}

/** Text blocks of a response `content` field (string, parts, or null). */
function textBlocksOf(
  raw: OpenAIResponse["choices"][number]["message"]["content"],
): ContentBlock[] {
  if (typeof raw === "string") {
    return raw.length > 0 ? [{ type: "text", text: raw }] : [];
  }
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((part) => part.type === "text" && part.text !== undefined)
    .map((part) => ({ type: "text", text: part.text! }));
}

/** Parse a tool call's JSON `arguments`; malformed JSON yields `{}`. */
function parseArguments(args: string | undefined): unknown {
  if (!args) return {};
  try {
    return JSON.parse(args);
  } catch {
    return {};
  }
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

  const content = collapseBlocks([
    ...textBlocksOf(choice.message.content),
    ...(choice.message.tool_calls ?? []).map((tc): ContentBlock => ({
      type: "tool_use",
      id: tc.id ?? "",
      name: tc.function.name ?? "",
      input: parseArguments(tc.function.arguments),
    })),
  ]);

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
          // The first fragment of a call carries id + name; later fragments
          // carry only argument text for the same index.
          if (tc.id || tc.function.name) {
            out.push({
              type: "tool_use",
              id: tc.id ?? "",
              name: tc.function.name ?? "",
            });
          }
          if (tc.function.arguments) {
            out.push({
              type: "tool_input_delta",
              id: tc.id ?? String(tc.index ?? 0),
              partial_json: tc.function.arguments,
            });
          }
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
