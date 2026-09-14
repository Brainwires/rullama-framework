/**
 * The Anthropic Messages wire format, shared by `AnthropicChatProvider`
 * (api.anthropic.com) and `BedrockProvider` (the same body inside an AWS
 * SigV4-signed request): message and tool conversion, response parsing,
 * request-body construction and stream-event mapping.
 *
 * @module
 */

import {
  type ChatOptions,
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

/** A message in Anthropic wire format. */
export interface AnthropicMessage {
  role: string;
  content: AnthropicContentBlock[];
}

/** A content block in Anthropic wire format. */
export type AnthropicContentBlock =
  | { type: "text"; text: string }
  // deno-lint-ignore no-explicit-any
  | { type: "tool_use"; id: string; name: string; input: any }
  | { type: "tool_result"; tool_use_id: string; content: string };

/** A tool definition in Anthropic wire format. */
export interface AnthropicTool {
  name: string;
  description: string;
  input_schema: ReturnType<typeof toolInputJsonSchema>;
}

/** A response content block as received (fields may be absent on Bedrock). */
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
  message?: { usage?: { input_tokens?: number } };
  delta?: { text?: string };
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
      };
    default:
      return null;
  }
}

/** Convert core Messages to Anthropic wire format (system messages are lifted out). */
export function convertMessages(messages: Message[]): AnthropicMessage[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: typeof m.content === "string"
        ? [{ type: "text" as const, text: m.content }]
        : m.content.map(convertBlock).filter((b): b is AnthropicContentBlock =>
          b !== null
        ),
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

/** Core content for a list of response blocks (see `collapseBlocks`). */
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

/** Parse an {@link AnthropicResponse} into a core ChatResponse. */
export function parseAnthropicResponse(
  response: AnthropicResponse,
): ChatResponse {
  const usage: Usage = {
    prompt_tokens: response.usage.input_tokens,
    completion_tokens: response.usage.output_tokens,
    total_tokens: response.usage.input_tokens + response.usage.output_tokens,
  };
  return {
    message: new Message({
      role: "assistant",
      content: parseContentBlocks(response.content),
    }),
    usage,
    finish_reason: response.stop_reason,
  };
}

/**
 * Build a Messages request body. `extra` carries the endpoint-specific fields
 * (`model` + `stream` for the public API, `anthropic_version` for Bedrock).
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
  if (options.temperature !== undefined) body.temperature = options.temperature;
  if (options.top_p !== undefined) body.top_p = options.top_p;
  if (options.stop) body.stop_sequences = options.stop;
  if (tools && tools.length > 0) body.tools = convertTools(tools);
  return body;
}

/** Mutable per-stream state for {@link streamEventToChunks}. */
export interface AnthropicStreamState {
  /** Input tokens, reported once in `message_start`. */
  promptTokens: number;
}

/**
 * Map one stream event to zero or more core chunks. Usage is reported on
 * `message_delta` with the prompt tokens remembered from `message_start`.
 */
export function streamEventToChunks(
  event: AnthropicStreamEvent,
  state: AnthropicStreamState,
): StreamChunk[] {
  switch (event.type) {
    case "message_start":
      state.promptTokens = event.message?.usage?.input_tokens ?? 0;
      return [];
    case "content_block_delta":
      return event.delta?.text
        ? [{ type: "text", text: event.delta.text }]
        : [];
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
    case "message_stop":
      return [{ type: "done" }];
    default:
      return [];
  }
}
