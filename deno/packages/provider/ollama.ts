// deno-lint-ignore-file no-explicit-any
/**
 * Ollama local model chat provider implementation.
 * Uses fetch() to the local Ollama API.
 * Equivalent to Rust's `ollama/mod.rs` + `ollama/chat.rs`.
 */

import {
  type ChatOptions,
  type ChatResponse,
  type ContentBlock,
  Message,
  type Provider,
  type StreamChunk,
  type Tool,
  toolInputJsonSchema,
  type Usage,
} from "@rullama/core";
import { postJson } from "./http.ts";
import { collapseBlocks, textBlock, toolUseBlock } from "./content.ts";
import { parseNDJSONStream } from "./sse.ts";
import { withSystemPrompt } from "./system_prompt.ts";

const DEFAULT_OLLAMA_URL = "http://localhost:11434";

// ---------------------------------------------------------------------------
// Ollama wire types
// ---------------------------------------------------------------------------

interface OllamaMessage {
  role: string;
  content: string;
  images?: string[];
  /** Calls an assistant message made (native `/api/chat` form). */
  tool_calls?: OllamaToolCall[];
  /** On a `role: "tool"` message: the tool whose output this is. */
  tool_name?: string;
  /** On a `role: "tool"` message: the call it answers (newer servers). */
  tool_call_id?: string;
}

/** A native Ollama tool call: `arguments` is a JSON object, not a string;
 * `id` is only sent by newer servers. */
interface OllamaToolCall {
  id?: string;
  function: {
    index?: number;
    name: string;
    arguments?: Record<string, any> | string;
  };
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
  message: { content: string; tool_calls?: OllamaToolCall[] };
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
}

interface OllamaStreamChunk {
  message?: { content?: string; tool_calls?: OllamaToolCall[] };
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

  /**
   * Create a provider for a local Ollama server (no API key).
   *
   * @param model Ollama model tag (e.g. `llama3.3`).
   * @param baseUrl Server root (default: `http://localhost:11434`).
   */
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
    const response = await this.post(
      this.buildRequestBody(messages, tools, options, false),
      options.signal,
    );
    return parseOllamaResponse(await response.json() as OllamaResponse);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const body = this.buildRequestBody(messages, tools, options, true);
    const response = await this.post(body, options.signal);

    if (!response.body) {
      throw new Error("Ollama streaming response has no body");
    }

    let toolCallCount = 0;
    for await (const line of parseNDJSONStream(response.body)) {
      let chunk: OllamaStreamChunk;
      try {
        chunk = JSON.parse(line);
      } catch {
        continue;
      }

      const content = chunk.message?.content;
      if (content) yield { type: "text", text: content };

      for (const call of chunk.message?.tool_calls ?? []) {
        // Ollama streams each call whole (arguments already parsed), so one
        // call becomes a `tool_use` start plus a single `tool_input_delta`.
        const { id, name, input } = toolCallParts(call, toolCallCount++);
        yield { type: "tool_use", id, name };
        yield {
          type: "tool_input_delta",
          id,
          partial_json: JSON.stringify(input),
        };
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

  /** POST to `/api/chat`; aborting `signal` closes the connection, which
   * makes Ollama stop generating. Throws with the body on a non-2xx status. */
  private post(
    body: Record<string, any>,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    return postJson("Ollama", `${this.baseUrl}/api/chat`, {}, body, signal);
  }

  /** Build the `/api/chat` body: converted messages (with `options.system`
   * as the leading system message), `temperature` / `top_p` under `options`,
   * tools in Ollama's native function form, and the `stream` flag. */
  private buildRequestBody(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
    stream: boolean,
  ): Record<string, any> {
    const ollamaMessages = withSystemPrompt(
      convertMessages(messages),
      options.system,
      (content) => ({ role: "system", content }),
    );

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

type ToolUseBlock = Extract<ContentBlock, { type: "tool_use" }>;
type ToolResultBlock = Extract<ContentBlock, { type: "tool_result" }>;

/**
 * Convert core Messages to Ollama wire format.
 *
 * An assistant message's `tool_use` blocks become native `tool_calls`
 * (arguments as a JSON object); each `tool_result` block becomes its own
 * `role: "tool"` message carrying the tool's name (looked up from the call
 * it answers) and the call id.
 */
export function convertMessages(messages: Message[]): OllamaMessage[] {
  const toolNames = new Map<string, string>();
  return messages.flatMap((m) => convertMessage(m, toolNames));
}

/** A message's blocks, sorted by what they become on the wire. */
interface MessageParts {
  text: string[];
  images: string[];
  toolUses: ToolUseBlock[];
  toolResults: ToolResultBlock[];
}

/** Sort `blocks` into text, image data, tool calls and tool results. */
function partsOf(blocks: ContentBlock[]): MessageParts {
  const parts: MessageParts = {
    text: [],
    images: [],
    toolUses: [],
    toolResults: [],
  };
  for (const block of blocks) {
    if (block.type === "text") parts.text.push(block.text);
    else if (block.type === "image") parts.images.push(block.source.data);
    else if (block.type === "tool_use") parts.toolUses.push(block);
    else if (block.type === "tool_result") parts.toolResults.push(block);
  }
  return parts;
}

/** The `role: "tool"` message for one result; the name comes from the call
 * it answers, when that call is in the conversation. */
function toolResultMessage(
  r: ToolResultBlock,
  toolNames: Map<string, string>,
): OllamaMessage {
  const msg: OllamaMessage = {
    role: "tool",
    content: String(r.content ?? ""),
    tool_call_id: r.tool_use_id,
  };
  const name = toolNames.get(r.tool_use_id);
  if (name !== undefined) msg.tool_name = name;
  return msg;
}

/** The text/image/tool-call message for `parts`, or `null` when the message
 * holds only tool results. */
function primaryMessage(
  role: string,
  parts: MessageParts,
): OllamaMessage | null {
  const { text, images, toolUses, toolResults } = parts;
  const empty = text.length + images.length + toolUses.length === 0;
  if (empty && toolResults.length > 0) return null;

  const msg: OllamaMessage = {
    role: toolUses.length > 0 ? "assistant" : role,
    content: text.join("\n"),
  };
  if (images.length > 0) msg.images = images;
  if (toolUses.length > 0) {
    msg.tool_calls = toolUses.map((t) => ({
      id: t.id,
      function: { name: t.name, arguments: t.input ?? {} },
    }));
  }
  return msg;
}

/** Convert one core message to zero or more Ollama messages, recording the
 * name of every tool call it makes in `toolNames` (id → name). */
function convertMessage(
  m: Message,
  toolNames: Map<string, string>,
): OllamaMessage[] {
  if (typeof m.content === "string") {
    return [{ role: m.role, content: m.content }];
  }
  const parts = partsOf(m.content);
  for (const t of parts.toolUses) toolNames.set(t.id, t.name);
  const results = parts.toolResults.map((r) => toolResultMessage(r, toolNames));
  const primary = primaryMessage(m.role, parts);
  return primary ? [primary, ...results] : results;
}

/** Convert core Tools to Ollama wire format. */
export function convertTools(tools: Tool[]): OllamaTool[] {
  return tools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: toolInputJsonSchema(t.input_schema),
    },
  }));
}

/** Arguments as an object: native calls send one, but tolerate a JSON
 * string (malformed JSON yields `{}`). */
function argumentsOf(args: OllamaToolCall["function"]["arguments"]): unknown {
  if (args === undefined) return {};
  if (typeof args !== "string") return args;
  try {
    return JSON.parse(args);
  } catch {
    return {};
  }
}

/** Id, name and input of the `n`-th tool call in a response. Older Ollama
 * servers send no id, so one is synthesised (`call_<n>`) for the tool
 * result to reference. */
function toolCallParts(
  call: OllamaToolCall,
  n: number,
): { id: string; name: string; input: unknown } {
  return {
    id: call.id || `call_${n}`,
    name: call.function.name,
    input: argumentsOf(call.function.arguments),
  };
}

/**
 * Parse an Ollama response into a core ChatResponse.
 *
 * `message.tool_calls` become `tool_use` blocks, and `finish_reason` is
 * `"tool_calls"` when there are any (Ollama itself reports `"stop"`), so
 * agent loops run the tools instead of treating the turn as final.
 */
export function parseOllamaResponse(
  response: OllamaResponse,
): ChatResponse {
  const calls = response.message.tool_calls ?? [];
  const blocks: ContentBlock[] = [];
  if (response.message.content) {
    blocks.push(textBlock(response.message.content));
  }
  calls.forEach((call, n) => {
    const { id, name, input } = toolCallParts(call, n);
    blocks.push(toolUseBlock(id, name, input));
  });
  const promptTokens = response.prompt_eval_count ?? 0;
  const completionTokens = response.eval_count ?? 0;

  const usage: Usage = {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };

  return {
    message: new Message({
      role: "assistant",
      content: collapseBlocks(blocks),
    }),
    usage,
    finish_reason: calls.length > 0
      ? "tool_calls"
      : response.done_reason ?? "stop",
  };
}
