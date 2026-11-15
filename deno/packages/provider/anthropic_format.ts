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
import {
  firstSystemText,
  mapBlocks,
  textBlock,
  toolUseBlock,
} from "./content.ts";

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

/**
 * The `Message.metadata` key under which a reply's `stop_details` are kept
 * (e.g. `{ category, explanation }` when `finish_reason` is `"refusal"`).
 * Present only when the API sent non-null `stop_details`. Never sent back.
 */
export const ANTHROPIC_STOP_DETAILS_KEY = "anthropicStopDetails";

/** Why a reply stopped, beyond `stop_reason` (e.g. a refusal's category). */
export interface AnthropicStopDetails {
  /** E.g. the refusal category. */
  category?: string | null;
  /** A human-readable explanation, when the API gives one. */
  explanation?: string;
  /** Any other fields, as the API sent them. */
  [field: string]: unknown;
}

/** A non-streaming response in Anthropic wire format. */
export interface AnthropicResponse {
  content: AnthropicResponseBlock[];
  stop_reason: string;
  stop_details?: AnthropicStopDetails | null;
  usage: { input_tokens: number; output_tokens: number };
}

/** The reply metadata: raw blocks and stop details, each only when present. */
function replyMetadata(
  blocks: readonly unknown[],
  stopDetails: AnthropicStopDetails | null | undefined,
): Record<string, unknown> | undefined {
  const metadata: Record<string, unknown> = {};
  if (blocks.length > 0) metadata[ANTHROPIC_CONTENT_KEY] = blocks;
  if (stopDetails) metadata[ANTHROPIC_STOP_DETAILS_KEY] = stopDetails;
  return Object.keys(metadata).length > 0 ? metadata : undefined;
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
    /** On `message_delta`. */
    stop_reason?: string;
    /** On `message_delta`: e.g. a refusal's category. */
    stop_details?: AnthropicStopDetails | null;
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

/** A message's core content as wire blocks, empty text blocks left out. */
function wireContent(m: Message): AnthropicContentBlock[] {
  const blocks = typeof m.content === "string"
    ? [{ type: "text" as const, text: m.content }]
    : m.content.map(convertBlock);
  return blocks.filter((b): b is AnthropicContentBlock =>
    b !== null && !(b.type === "text" && b.text === "")
  );
}

/**
 * Convert core Messages to Anthropic wire format.
 *
 * - System messages are lifted out (the system prompt goes in `system`).
 * - An assistant message with raw reply blocks in
 *   `metadata[ANTHROPIC_CONTENT_KEY]` is sent as those blocks, unchanged, as
 *   a message of its own (never merged or altered).
 * - Empty text blocks are left out, and a message left with no content is
 *   dropped (the API refuses both).
 * - Consecutive user turns (`user` and `tool` messages, both sent as
 *   `user`) are merged into one, so the results of parallel tool calls
 *   arrive together. Assistant turns are not merged.
 */
export function convertMessages(messages: Message[]): AnthropicMessage[] {
  const out: AnthropicMessage[] = [];
  let lastMergeable = false;
  for (const m of messages) {
    if (m.role === "system") continue;
    const stored = storedBlocks(m);
    if (stored) {
      out.push({ role: "assistant", content: stored });
      lastMergeable = false;
      continue;
    }
    const content = wireContent(m);
    if (content.length === 0) continue;
    const role = m.role === "assistant" ? "assistant" : "user";
    const last = out.at(-1);
    if (role === "user" && lastMergeable && last) {
      last.content = [...last.content, ...content];
    } else {
      out.push({ role, content });
    }
    lastMergeable = role === "user";
  }
  return out;
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
  return firstSystemText(messages);
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
 * carries the reply's raw blocks in `metadata[ANTHROPIC_CONTENT_KEY]` and its
 * `stop_details`, if any, in `metadata[ANTHROPIC_STOP_DETAILS_KEY]`.
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
  const metadata = replyMetadata(blocks, response.stop_details);
  return {
    message: new Message({
      role: "assistant",
      content: parseContentBlocks(blocks),
      ...(metadata ? { metadata } : {}),
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
  /** The reply's `stop_details`, from `message_delta`. */
  stopDetails?: AnthropicStopDetails | null;
}

type Delta = NonNullable<AnthropicStreamEvent["delta"]>;
type DeltaHandler = (
  block: AnthropicRawBlock,
  delta: Delta,
  index: number,
  state: AnthropicStreamState,
) => StreamChunk[];
type EventHandler = (
  event: AnthropicStreamEvent,
  state: AnthropicStreamState,
) => StreamChunk[];

/** Append `more` to the string field `field` of `block`. */
function append(
  block: AnthropicRawBlock,
  field: string,
  more: string | undefined,
): void {
  block[field] = String(block[field] ?? "") + (more ?? "");
}

/** `text_delta` (or an untyped delta with text): extend the block, stream the text. */
const textDelta: DeltaHandler = (block, delta) => {
  if (!delta.text) return [];
  append(block, "text", delta.text);
  return [{ type: "text", text: delta.text }];
};

/** Tool input JSON: gathered per block (parsed at `content_block_stop`) and streamed. */
const inputJsonDelta: DeltaHandler = (block, delta, index, state) => {
  const part = delta.partial_json ?? "";
  const json = (state.partialJson ??= {});
  json[index] = (json[index] ?? "") + part;
  if (block.type !== "tool_use" || !part) return [];
  return [{
    type: "tool_input_delta",
    id: String(block.id ?? ""),
    partial_json: part,
  }];
};

/** Thinking and signatures are kept in the block only, never streamed as text. */
const DELTA_HANDLERS = new Map<string, DeltaHandler>([
  ["text_delta", textDelta],
  ["input_json_delta", inputJsonDelta],
  ["thinking_delta", (block, delta) => {
    append(block, "thinking", delta.thinking);
    return [];
  }],
  ["signature_delta", (block, delta) => {
    append(block, "signature", delta.signature);
    return [];
  }],
  ["citations_delta", (block, delta) => {
    const citations = Array.isArray(block.citations) ? block.citations : [];
    block.citations = [...citations, delta.citation];
    return [];
  }],
]);

/** `content_block_delta`: apply the delta to its block; the chunks it yields. */
const applyDelta: EventHandler = (event, state) => {
  const delta = event.delta;
  if (!delta) return [];
  const index = event.index ?? 0;
  const block = state.blocks?.[index];
  // A delta for a block that was never started still streams its text.
  if (!block) return delta.text ? [{ type: "text", text: delta.text }] : [];
  const handler = DELTA_HANDLERS.get(delta.type ?? "text_delta") ?? textDelta;
  return handler(block, delta, index, state);
};

/** `content_block_start`: open the block; announce a tool call. */
const startBlock: EventHandler = (event, state) => {
  const start = event.content_block;
  if (!start) return [];
  (state.blocks ??= [])[event.index ?? 0] = { ...start };
  return startChunks(start);
};

/** The chunks a block's start yields: a tool call's id and name, or initial text. */
function startChunks(block: AnthropicResponseBlock): StreamChunk[] {
  if (block.type === "tool_use") {
    return [{ type: "tool_use", id: block.id ?? "", name: block.name ?? "" }];
  }
  return block.type === "text" && block.text
    ? [{ type: "text", text: block.text }]
    : [];
}

/** `content_block_stop`: parse the block's gathered tool input JSON. */
const stopBlock: EventHandler = (event, state) => {
  const index = event.index ?? 0;
  const pending = state.partialJson ?? {};
  const json = pending[index];
  const block = state.blocks?.[index];
  if (json === undefined || !block) return [];
  delete pending[index];
  try {
    block.input = json.trim() ? JSON.parse(json) : {};
  } catch {
    // Keep the start block's input when the gathered JSON is malformed.
  }
  return [];
};

/** `message_delta`: remember `stop_details`; usage, with the prompt tokens
 * from `message_start`. */
const usageChunk: EventHandler = (event, state) => {
  if (event.delta?.stop_details) state.stopDetails = event.delta.stop_details;
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
};

/** `message_stop`: the reply's raw blocks (and stop details) as a `metadata`
 * chunk, then `done`. */
const stopMessage: EventHandler = (_event, state) => {
  const blocks = (state.blocks ?? []).filter((b) => b !== undefined);
  const metadata = replyMetadata(blocks, state.stopDetails);
  const done: StreamChunk = { type: "done" };
  return metadata ? [{ type: "metadata", metadata }, done] : [done];
};

const EVENT_HANDLERS = new Map<string, EventHandler>([
  ["message_start", (event, state) => {
    state.promptTokens = event.message?.usage?.input_tokens ?? 0;
    state.blocks = [];
    state.partialJson = {};
    return [];
  }],
  ["content_block_start", startBlock],
  ["content_block_delta", applyDelta],
  ["content_block_stop", stopBlock],
  ["message_delta", usageChunk],
  ["message_stop", stopMessage],
]);

/**
 * Map one stream event to zero or more core chunks. Usage is reported on
 * `message_delta` with the prompt tokens remembered from `message_start`.
 *
 * The reply's raw blocks are gathered in `state` (text, thinking, signature
 * and citation deltas, tool input JSON) and, on `message_stop`, sent as a
 * `metadata` chunk (`{ [ANTHROPIC_CONTENT_KEY]: blocks }`, plus
 * `[ANTHROPIC_STOP_DETAILS_KEY]` when `message_delta` carried `stop_details`)
 * just before `done`: store it on the assembled assistant message so the next
 * request sends the reply back unchanged.
 */
export function streamEventToChunks(
  event: AnthropicStreamEvent,
  state: AnthropicStreamState,
): StreamChunk[] {
  return EVENT_HANDLERS.get(event.type)?.(event, state) ?? [];
}
