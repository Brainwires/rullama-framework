// deno-lint-ignore-file no-explicit-any
/**
 * OpenAI Responses API provider implementation.
 * Uses the newer `/v1/responses` endpoint (superseding Chat Completions).
 * Equivalent to Rust's `openai_responses/` module.
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

const DEFAULT_BASE_URL = "https://api.openai.com/v1/responses";

// ---------------------------------------------------------------------------
// Wire types — Responses API input
// ---------------------------------------------------------------------------

/** Input content: plain string or structured parts. */
type InputContent = string | InputContentPart[];

interface InputContentPart {
  type: string;
  text?: string;
  image_url?: string;
}

/** An input item for the Responses API. */
interface ResponseInputItem {
  type: string;
  role?: string;
  content?: InputContent;
  status?: string;
  call_id?: string;
  output?: string;
}

/** Tool definition for the Responses API. */
interface ResponseTool {
  type: "function";
  name: string;
  description: string;
  parameters: Record<string, any>;
  strict?: boolean;
}

// ---------------------------------------------------------------------------
// Wire types — Responses API output
// ---------------------------------------------------------------------------

interface OutputContentBlock {
  type: string;
  text?: string;
  refusal?: string;
  transcript?: string;
  annotations?: any[];
}

interface ResponseOutputItem {
  type: string;
  id?: string;
  role?: string;
  content?: OutputContentBlock[];
  status?: string;
  name?: string;
  arguments?: string;
  call_id?: string;
}

interface ResponseUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens?: number;
}

interface ResponseObject {
  id: string;
  object?: string;
  status?: string;
  output: ResponseOutputItem[];
  output_text?: string;
  usage?: ResponseUsage;
  [key: string]: any;
}

// ---------------------------------------------------------------------------
// Wire types — Responses API streaming
// ---------------------------------------------------------------------------

interface ResponseStreamEvent {
  type: string;
  delta?: string;
  item_id?: string;
  item?: ResponseOutputItem;
  output_index?: number;
  content_index?: number;
  response?: ResponseObject;
  [key: string]: any;
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Convert core Messages to Responses API input items.
 * Returns [items, systemPrompt]. */
export function messagesToInput(
  messages: Message[],
): [ResponseInputItem[], string | undefined] {
  const items: ResponseInputItem[] = [];
  let systemPrompt: string | undefined;

  for (const msg of messages) {
    switch (msg.role) {
      case "system":
        if (typeof msg.content === "string") {
          systemPrompt = msg.content;
        }
        break;
      case "user":
      case "assistant": {
        if (typeof msg.content === "string") {
          items.push({
            type: "message",
            role: msg.role,
            content: msg.content,
          });
        } else {
          // Check for tool results in blocks
          for (const block of msg.content) {
            if (block.type === "text") {
              items.push({
                type: "message",
                role: msg.role,
                content: block.text,
              });
            } else if (block.type === "tool_result") {
              items.push({
                type: "function_call_output",
                call_id: block.tool_use_id,
                output: block.content,
              });
            }
          }
        }
        break;
      }
      case "tool": {
        const text = typeof msg.content === "string" ? msg.content : undefined;
        if (text) {
          items.push({
            type: "function_call_output",
            call_id: msg.name ?? "",
            output: text,
          });
        }
        break;
      }
    }
  }

  return [items, systemPrompt];
}

/** Convert core Tools to Responses API function tool definitions. */
export function toolsToResponseTools(tools: Tool[]): ResponseTool[] {
  return tools.map((t) => ({
    type: "function" as const,
    name: t.name,
    description: t.description,
    parameters: t.input_schema.properties ?? {},
  }));
}

/** Parse a ResponseObject into a core ChatResponse. */
export function responseToChat(resp: ResponseObject): ChatResponse {
  const contentBlocks: ContentBlock[] = [];

  for (const item of resp.output) {
    if (item.type === "message" && item.content) {
      for (const block of item.content) {
        if (block.type === "output_text" && block.text !== undefined) {
          contentBlocks.push({ type: "text", text: block.text });
        } else if (block.type === "refusal" && block.refusal !== undefined) {
          contentBlocks.push({ type: "text", text: block.refusal });
        } else if (block.type === "output_audio" && block.transcript) {
          contentBlocks.push({ type: "text", text: block.transcript });
        }
      }
    } else if (item.type === "function_call") {
      let input: any = {};
      try {
        input = JSON.parse(item.arguments ?? "{}");
      } catch { /* use empty object */ }
      contentBlocks.push({
        type: "tool_use",
        id: item.call_id ?? "",
        name: item.name ?? "",
        input,
      });
    }
  }

  let content: MessageContent;
  if (contentBlocks.length === 1 && contentBlocks[0].type === "text") {
    content = contentBlocks[0].text;
  } else if (contentBlocks.length === 0) {
    content = "";
  } else {
    content = contentBlocks;
  }

  const usage = convertUsage(resp.usage);

  return {
    message: new Message({ role: "assistant", content }),
    usage,
    finish_reason: "stop",
  };
}

/** Convert ResponseUsage to core Usage. */
export function convertUsage(usage?: ResponseUsage): Usage {
  if (!usage) {
    return { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  }
  return {
    prompt_tokens: usage.input_tokens,
    completion_tokens: usage.output_tokens,
    total_tokens: usage.total_tokens ??
      (usage.input_tokens + usage.output_tokens),
  };
}

/** Convert a streaming event to StreamChunk values. Returns undefined for events we skip. */
export function streamEventToChunks(
  event: ResponseStreamEvent,
): StreamChunk[] | undefined {
  switch (event.type) {
    case "response.output_text.delta":
      if (event.delta !== undefined) {
        return [{ type: "text", text: event.delta }];
      }
      return undefined;

    case "response.output_item.added":
      if (event.item?.type === "function_call") {
        return [{
          type: "tool_use",
          id: event.item.call_id ?? "",
          name: event.item.name ?? "",
        }];
      }
      return undefined;

    case "response.function_call_arguments.delta":
      if (event.delta !== undefined && event.item_id !== undefined) {
        return [{
          type: "tool_input_delta",
          id: event.item_id,
          partial_json: event.delta,
        }];
      }
      return undefined;

    case "response.completed":
      if (event.response) {
        const usage = convertUsage(event.response.usage);
        return [{ type: "usage", usage }, { type: "done" }];
      }
      return [{ type: "done" }];

    case "response.failed":
    case "response.incomplete":
      return [{ type: "done" }];

    default:
      return undefined;
  }
}

/** Build a request body for POST /v1/responses. */
export function buildRequestBody(
  model: string,
  input: ResponseInputItem[],
  instructions: string | undefined,
  tools: ResponseTool[] | undefined,
  options: ChatOptions,
  previousResponseId?: string,
): Record<string, any> {
  const body: Record<string, any> = {
    model,
    input,
  };

  if (instructions) body.instructions = instructions;
  if (tools && tools.length > 0) {
    body.tools = tools;
    body.tool_choice = "auto";
  }
  if (options.max_tokens !== undefined) {
    body.max_output_tokens = options.max_tokens;
  }
  if (options.temperature !== undefined) body.temperature = options.temperature;
  if (options.top_p !== undefined) body.top_p = options.top_p;
  if (options.stop) body.stop = options.stop;
  if (previousResponseId) body.previous_response_id = previousResponseId;

  return body;
}

// ---------------------------------------------------------------------------
// OpenAiResponsesProvider
// ---------------------------------------------------------------------------

/** Chat provider backed by the OpenAI Responses API.
 * Tracks the last response ID for automatic conversation chaining.
 * Equivalent to Rust's `OpenAiResponsesProvider`. */
export class OpenAiResponsesProvider implements Provider {
  readonly name: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;
  private lastResponseId: string | undefined;

  constructor(
    apiKey: string,
    model: string,
    baseUrl?: string,
    providerName?: string,
  ) {
    this.apiKey = apiKey;
    this.model = model;
    this.baseUrl = baseUrl ?? DEFAULT_BASE_URL;
    this.name = providerName ?? "openai-responses";
  }

  /** Create a copy with a different provider name. */
  withProviderName(name: string): OpenAiResponsesProvider {
    return new OpenAiResponsesProvider(
      this.apiKey,
      this.model,
      this.baseUrl,
      name,
    );
  }

  /** Get the last response ID (for manual chaining). */
  getLastResponseId(): string | undefined {
    return this.lastResponseId;
  }

  // -----------------------------------------------------------------------
  // Provider interface
  // -----------------------------------------------------------------------

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    const [input, system] = messagesToInput(messages);
    const responseTools = tools ? toolsToResponseTools(tools) : undefined;
    const instructions = system ?? options.system;

    const body = buildRequestBody(
      this.model,
      input,
      instructions,
      responseTools,
      options,
      this.lastResponseId,
    );

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
        `Responses API error (${response.status}): ${errorText}`,
      );
    }

    const resp: ResponseObject = await response.json();
    this.lastResponseId = resp.id;

    return responseToChat(resp);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const [input, system] = messagesToInput(messages);
    const responseTools = tools ? toolsToResponseTools(tools) : undefined;
    const instructions = system ?? options.system;

    const body = buildRequestBody(
      this.model,
      input,
      instructions,
      responseTools,
      options,
      this.lastResponseId,
    );
    body.stream = true;

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
        `Responses API error (${response.status}): ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error("Responses API streaming response has no body");
    }

    for await (const data of parseSSEStream(response.body)) {
      let event: ResponseStreamEvent;
      try {
        event = JSON.parse(data);
      } catch {
        continue;
      }

      // Store response ID from completed events
      if (event.type === "response.completed" && event.response) {
        this.lastResponseId = event.response.id;
      }

      const chunks = streamEventToChunks(event);
      if (chunks) {
        for (const chunk of chunks) {
          yield chunk;
        }
      }
    }
  }
}
