/**
 * The Anthropic Messages wire format, shared by `AnthropicChatProvider`
 * (api.anthropic.com) and `BedrockProvider` (the same body inside an AWS
 * SigV4-signed request): message and tool conversion, response parsing,
 * request-body construction and stream-event mapping.
 *
 * Replies keep their raw content blocks (thinking and redacted thinking
 * included) in `Message.metadata[ANTHROPIC_CONTENT_KEY]`, and an assistant
 * message carrying them is sent back exactly as it came: current models
 * require their thinking blocks back unchanged, in order.
 *
 * @module
 */

import {
  ChatOptions,
  type ChatResponse,
  type ContentBlock,
  Message,
  type MessageContent,
  type StreamChunk,
  type Tool,
  toolInputJsonSchema,
  type Usage,
} from "@rullama/core";
import { mapBlocks, textBlock, toolUseBlock } from "./content.ts";

/**
 * The `Message.metadata` key under which a reply's raw content blocks are
 * kept (`{ anthropicContent: AnthropicRawBlock[] }`). When an assistant
 * message has them, they are sent instead of its `content`.
 */
export const ANTHROPIC_CONTENT_KEY = "anthropicContent";

/** A message in Anthropic wire format. */
export interface AnthropicMessage {
  role: string;
  content: (AnthropicContentBlock | AnthropicRawBlock)[];
}

/** A content block in Anthropic wire format. */
export type AnthropicContentBlock =
  | { type: "text"; text: string }
  // deno-lint-ignore no-explicit-any
  | { type: "tool_use"; id: string; name: string; input: any }
  | {
    type: "tool_result";
    tool_use_id: string;
    content: string;
    is_error?: boolean;
  };

/**
 * A content block exactly as the API sent it: `text`, `tool_use`, `thinking`
 * (`thinking`, `signature`), `redacted_thinking` (`data`), or any other kind.
 */
export interface AnthropicRawBlock {
  /** The block kind (`text`, `tool_use`, `thinking`, `redacted_thinking`, …). */
  type: string;
  /** The kind's own fields, as the API sent them. */
  [field: string]: unknown;
}

/** A tool definition in Anthropic wire format. */
export interface AnthropicTool {
  name: string;
  description: string;
  input_schema: ReturnType<typeof toolInputJsonSchema>;
}

/**
 * A response content block as received (fields may be absent on Bedrock).
 * Only the fields the core content is built from are typed; the block's
 * other fields (e.g. `thinking`, `signature`, `data`) are kept as they came.
 */
export interface AnthropicResponseBlock {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  // deno-lint-ignore no-explicit-any
  input?: any;
}

/** A non-streaming response in Anthropic wire format. */
export interface AnthropicResponse {
  content: AnthropicResponseBlock[];
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number };
}

/** A streaming event in Anthropic wire format (the subset we consume). */
export interface AnthropicStreamEvent {
  type: string;
  /** The content block a `content_block_*` event is about. */
  index?: number;
  message?: { usage?: { input_tokens?: number } };
  /** The block a `content_block_start` opens. */
  content_block?: AnthropicResponseBlock;
  delta?: {
    /** `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`, `citations_delta`. */
    type?: string;
    text?: string;
    thinking?: string;
    signature?: string;
    partial_json?: string;
    citation?: unknown;
  };
  usage?: { input_tokens?: number; output_tokens?: number };
}

/** Convert one core content block to the wire format; `null` for unsupported kinds. */
function convertBlock(block: ContentBlock): AnthropicContentBlock | null {
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
        ...(block.is_error ? { is_error: true } : {}),
      };
    default:
      return null;
  }
}

/** The raw reply blocks kept on an assistant message, if it has any. */
function storedBlocks(m: Message): AnthropicRawBlock[] | undefined {
  if (m.role !== "assistant") return undefined;
  const raw = (m.metadata as Record<string, unknown> | undefined)?.[
    ANTHROPIC_CONTENT_KEY
  ];
  return Array.isArray(raw) && raw.length > 0
    ? raw as AnthropicRawBlock[]
    : undefined;
}

/**
 * Convert core Messages to Anthropic wire format (system messages are lifted
 * out). An assistant message with raw reply blocks in
 * `metadata[ANTHROPIC_CONTENT_KEY]` is sent as those blocks, unchanged.
 */
export function convertMessages(messages: Message[]): AnthropicMessage[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: storedBlocks(m) ??
        (typeof m.content === "string"
          ? [{ type: "text" as const, text: m.content }]
          : m.content.map(convertBlock).filter((
            b,
          ): b is AnthropicContentBlock => b !== null)),
    }));
}

/** Convert core Tools to Anthropic wire format. */
export function convertTools(tools: Tool[]): AnthropicTool[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: toolInputJsonSchema(t.input_schema),
  }));
}

/** Extract the first system message from the message list. */
export function getSystemMessage(messages: Message[]): string | undefined {
  const sys = messages.find((m) => m.role === "system");
  if (!sys) return undefined;
  return typeof sys.content === "string" ? sys.content : undefined;
}

/**
 * Core content for a list of response blocks (see `collapseBlocks`): text and
 * tool calls. Thinking blocks stay in the raw blocks only.
 */
export function parseContentBlocks(
  blocks: AnthropicResponseBlock[],
): MessageContent {
  return mapBlocks(
    blocks,
    (b) =>
      b.type === "text"
        ? textBlock(b.text ?? "")
        : b.type === "tool_use"
        ? toolUseBlock(b.id, b.name, b.input)
        : null,
  );
}

/**
 * Parse an {@link AnthropicResponse} into a core ChatResponse. The message
 * carries the reply's raw blocks in `metadata[ANTHROPIC_CONTENT_KEY]`.
 */
export function parseAnthropicResponse(
  response: AnthropicResponse,
): ChatResponse {
  const usage: Usage = {
    prompt_tokens: response.usage.input_tokens,
    completion_tokens: response.usage.output_tokens,
    total_tokens: response.usage.input_tokens + response.usage.output_tokens,
  };
  const blocks = Array.isArray(response.content) ? response.content : [];
  return {
    message: new Message({
      role: "assistant",
      content: parseContentBlocks(blocks),
      ...(blocks.length > 0
        ? { metadata: { [ANTHROPIC_CONTENT_KEY]: blocks } }
        : {}),
    }),
    usage,
    finish_reason: response.stop_reason,
  };
}

/**
 * Build a Messages request body. `extra` carries the endpoint-specific fields
 * (`model` + `stream` for the public API, `anthropic_version` for Bedrock).
 *
 * Sampling options (`temperature`, `top_p`, `top_k`) are sent only when the
 * caller set them: current models refuse them, and `ChatOptions`' default
 * temperature does not count (see `ChatOptions.explicitTemperature`).
 */
export function buildAnthropicBody(
  messages: Message[],
  tools: Tool[] | undefined,
  options: ChatOptions,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  const system = options.system ?? getSystemMessage(messages);
  const body: Record<string, unknown> = {
    ...extra,
    messages: convertMessages(messages),
    max_tokens: options.max_tokens ?? 4096,
  };
  if (system) body.system = system;
  const temperature = ChatOptions.explicitTemperature(options);
  if (temperature !== undefined) body.temperature = temperature;
  if (options.top_p !== undefined) body.top_p = options.top_p;
  if (options.top_k !== undefined) body.top_k = options.top_k;
  if (options.stop) body.stop_sequences = options.stop;
  if (tools && tools.length > 0) body.tools = convertTools(tools);
  return body;
}

/** Mutable per-stream state for {@link streamEventToChunks}. */
export interface AnthropicStreamState {
  /** Input tokens, reported once in `message_start`. */
  promptTokens: number;
  /** The reply's raw blocks so far, by index (thinking and signatures included). */
  blocks?: AnthropicRawBlock[];
  /** Tool-call input JSON gathered so far, by block index. */
  partialJson?: Record<number, string>;
}

/** Apply one `content_block_delta` to its block; the chunks it yields. */
function applyDelta(
  event: AnthropicStreamEvent,
  state: AnthropicStreamState,
): StreamChunk[] {
  const delta = event.delta;
  if (!delta) return [];
  const index = event.index ?? 0;
  const block = state.blocks?.[index];
  switch (delta.type) {
    case "thinking_delta":
      if (block) {
        block.thinking = String(block.thinking ?? "") + (delta.thinking ?? "");
      }
      return [];
    case "signature_delta":
      if (block) {
        block.signature = String(block.signature ?? "") +
          (delta.signature ?? "");
      }
      return [];
    case "input_json_delta": {
      const part = delta.partial_json ?? "";
      const json = (state.partialJson ??= {});
      json[index] = (json[index] ?? "") + part;
      return block?.type === "tool_use" && part
        ? [{
          type: "tool_input_delta",
          id: String(block.id ?? ""),
          partial_json: part,
        }]
        : [];
    }
    case "citations_delta":
      if (block) {
        const citations = Array.isArray(block.citations) ? block.citations : [];
        block.citations = [...citations, delta.citation];
      }
      return [];
    default: {
      // `text_delta` (or an untyped delta with text).
      if (!delta.text) return [];
      if (block) block.text = String(block.text ?? "") + delta.text;
      return [{ type: "text", text: delta.text }];
    }
  }
}

/**
 * Map one stream event to zero or more core chunks. Usage is reported on
 * `message_delta` with the prompt tokens remembered from `message_start`.
 *
 * The reply's raw blocks are gathered in `state` (text, thinking and
 * signature deltas, tool input JSON) and, on `message_stop`, sent as a
 * `metadata` chunk (`{ [ANTHROPIC_CONTENT_KEY]: blocks }`) just before `done`:
 * store it on the assembled assistant message so the next request sends the
 * reply back unchanged.
 */
export function streamEventToChunks(
  event: AnthropicStreamEvent,
  state: AnthropicStreamState,
): StreamChunk[] {
  switch (event.type) {
    case "message_start":
      state.promptTokens = event.message?.usage?.input_tokens ?? 0;
      state.blocks = [];
      state.partialJson = {};
      return [];
    case "content_block_start": {
      const start = event.content_block;
      if (!start) return [];
      const index = event.index ?? 0;
      const block: AnthropicRawBlock = { ...start };
      (state.blocks ??= [])[index] = block;
      if (block.type === "tool_use") {
        return [{
          type: "tool_use",
          id: String(block.id ?? ""),
          name: String(block.name ?? ""),
        }];
      }
      return block.type === "text" && typeof block.text === "string" &&
          block.text
        ? [{ type: "text", text: block.text }]
        : [];
    }
    case "content_block_delta":
      return applyDelta(event, state);
    case "content_block_stop": {
      const index = event.index ?? 0;
      const pending = state.partialJson ?? {};
      const json = pending[index];
      const block = state.blocks?.[index];
      if (json !== undefined && block) {
        try {
          block.input = json.trim() ? JSON.parse(json) : {};
        } catch {
          // Keep the start block's input when the gathered JSON is malformed.
        }
        delete pending[index];
      }
      return [];
    }
    case "message_delta": {
      if (!event.usage) return [];
      const completion = event.usage.output_tokens ?? 0;
      return [{
        type: "usage",
        usage: {
          prompt_tokens: state.promptTokens,
          completion_tokens: completion,
          total_tokens: state.promptTokens + completion,
        },
      }];
    }
    case "message_stop": {
      const blocks = (state.blocks ?? []).filter((b) => b !== undefined);
      return blocks.length > 0
        ? [
          {
            type: "metadata",
            metadata: { [ANTHROPIC_CONTENT_KEY]: blocks },
          },
          { type: "done" },
        ]
        : [{ type: "done" }];
    }
    default:
      return [];
  }
}
