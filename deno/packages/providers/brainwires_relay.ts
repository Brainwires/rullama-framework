/**
 * HTTP-based Brainwires Relay provider.
 *
 * Connects to the Brainwires Studio backend which routes requests to the
 * appropriate upstream model (Claude, GPT, Gemini, …). Uses Server-Sent
 * Events for streaming. Implements the `Provider` interface from
 * `@brainwires/core`.
 *
 * Equivalent to Rust's `brainwires_providers::brainwires_http` module.
 */

import type {
  ChatResponse,
  ContentBlock,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@brainwires/core";
import type { ChatOptions } from "@brainwires/core";
import { createUsage } from "@brainwires/core";
import { Message as MessageClass } from "@brainwires/core";

/** Production backend URL. */
export const DEFAULT_BACKEND_URL = "https://brainwires.studio";

/** Development backend URL. */
export const DEV_BACKEND_URL = "https://dev.brainwires.net";

/**
 * Determine the backend URL from an API key prefix.
 *
 * Keys starting with `bw_dev_` route to the dev backend; all others
 * (including `bw_prod_` and `bw_test_`) route to production.
 */
export function getBackendFromApiKey(api_key: string): string {
  return api_key.startsWith("bw_dev_") ? DEV_BACKEND_URL : DEFAULT_BACKEND_URL;
}

/** Per-model max output tokens lookup. */
export function maxOutputTokensForModel(model: string): number {
  if (model.includes("claude-3-5-sonnet")) return 8192;
  if (model.includes("claude-3-opus")) return 4096;
  if (model.includes("claude-3-haiku")) return 4096;
  if (model.includes("claude")) return 4096;
  if (model.includes("gpt-5")) return 32768;
  if (model.includes("gpt-4")) return 8192;
  if (model.includes("gpt-3.5")) return 4096;
  if (model.includes("o1")) return 65536;
  if (model.includes("gemini-1.5-pro")) return 8192;
  if (model.includes("gemini-1.5-flash")) return 8192;
  if (model.includes("gemini")) return 2048;
  return 8192;
}

/**
 * HTTP provider targeting the Brainwires Studio backend.
 */
export class BrainwiresRelayProvider implements Provider {
  readonly name = "brainwires";
  readonly backend_url: string;
  readonly model: string;
  private readonly api_key: string;

  constructor(api_key: string, backend_url: string, model: string) {
    this.api_key = api_key;
    this.backend_url = backend_url;
    this.model = model;
  }

  maxOutputTokens(): number {
    return maxOutputTokensForModel(this.model);
  }

  /** Exposed for tests — find the system prompt in a message list. */
  getSystemMessage(messages: Message[]): string | undefined {
    const sys = messages.find((m) => m.role === "system");
    return sys?.text();
  }

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    let full_text = "";
    let usage_data: ReturnType<typeof createUsage> | undefined;
    const tool_calls: ContentBlock[] = [];
    let last_response_id: string | undefined;

    for await (const chunk of this.streamChat(messages, tools, options)) {
      switch (chunk.type) {
        case "text":
          full_text += chunk.text;
          break;
        case "usage":
          usage_data = chunk.usage;
          break;
        case "done":
          break;
        case "tool_call":
          last_response_id = chunk.response_id;
          tool_calls.push({
            type: "tool_use",
            id: chunk.call_id,
            name: chunk.tool_name,
            input: chunk.parameters,
          });
          break;
        // tool_use / tool_input_delta are not emitted by the relay backend.
      }
    }

    const content: string | ContentBlock[] = tool_calls.length === 0
      ? full_text
      : (() => {
        const blocks: ContentBlock[] = [];
        if (full_text.length > 0) blocks.push({ type: "text", text: full_text });
        blocks.push(...tool_calls);
        return blocks;
      })();

    const message = new MessageClass({
      role: "assistant",
      content,
      metadata: last_response_id === undefined
        ? undefined
        : { response_id: last_response_id },
    });

    return {
      message,
      usage: usage_data ?? createUsage(0, 0),
      finish_reason: tool_calls.length === 0 ? "stop" : undefined,
    };
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    if (messages.length === 0) {
      throw new Error("No messages provided");
    }

    const { current_content, conversation_history, function_call_output, previous_response_id } =
      buildRequestParts(messages);

    const request_body: Record<string, unknown> = {
      content: current_content,
      model: this.model,
      timezone: "UTC",
    };
    if (conversation_history.length > 0) {
      request_body.conversationHistory = conversation_history;
    }
    if (function_call_output !== null) {
      request_body.functionCallOutput = function_call_output;
      if (previous_response_id !== null) {
        request_body.previousResponseId = previous_response_id;
      }
    }
    if (options.system !== undefined) request_body.systemPrompt = options.system;
    if (options.temperature !== undefined) request_body.temperature = options.temperature;

    if (tools && tools.length > 0) {
      request_body.selectedMCPTools = tools.map((t) => ({
        name: t.name,
        server: "cli-local",
        description: t.description,
        inputSchema: t.input_schema,
      }));
    }

    const url = `${this.backend_url}/api/chat/stream`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request_body),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "Unknown error");
      throw new Error(`Brainwires API error (${res.status}): ${body}`);
    }

    if (res.body === null) {
      yield { type: "done" };
      return;
    }

    for await (const event of parseSseEvents(res.body)) {
      const { eventType, data } = event;
      switch (eventType) {
        case "delta": {
          try {
            const parsed = JSON.parse(data) as { delta?: string };
            if (typeof parsed.delta === "string") {
              yield { type: "text", text: parsed.delta };
            }
          } catch {
            // ignore malformed deltas
          }
          break;
        }
        case "complete": {
          try {
            const parsed = JSON.parse(data) as { usage?: unknown };
            if (parsed.usage && typeof parsed.usage === "object") {
              const u = parsed.usage as {
                prompt_tokens?: number;
                completion_tokens?: number;
                total_tokens?: number;
              };
              const prompt = u.prompt_tokens ?? 0;
              const completion = u.completion_tokens ?? 0;
              yield {
                type: "usage",
                usage: {
                  prompt_tokens: prompt,
                  completion_tokens: completion,
                  total_tokens: u.total_tokens ?? prompt + completion,
                },
              };
            }
          } catch {
            // ignore malformed complete
          }
          yield { type: "done" };
          break;
        }
        case "error": {
          let msg = "Unknown error";
          try {
            const parsed = JSON.parse(data) as { message?: string };
            if (typeof parsed.message === "string") msg = parsed.message;
          } catch {
            // keep default message
          }
          throw new Error(`Stream error: ${msg}`);
        }
        case "toolCall": {
          try {
            const parsed = JSON.parse(data) as {
              callId?: string;
              responseId?: string;
              chatId?: string;
              toolName?: string;
              server?: string;
              parameters?: unknown;
            };
            const call_id = typeof parsed.callId === "string" ? parsed.callId : "";
            const response_id = typeof parsed.responseId === "string" ? parsed.responseId : "";
            const tool_name = typeof parsed.toolName === "string" ? parsed.toolName : "";
            const server = typeof parsed.server === "string" ? parsed.server : "";
            const chat_id = typeof parsed.chatId === "string" ? parsed.chatId : undefined;
            const parameters = parsed.parameters ?? {};
            yield {
              type: "tool_call",
              call_id,
              response_id,
              chat_id,
              tool_name,
              server,
              parameters,
            };
            yield { type: "done" };
            return;
          } catch {
            // ignore malformed toolCall
          }
          break;
        }
        case "title":
          // Ignored.
          break;
        default:
          // Unknown event — ignored for forward compatibility.
          break;
      }
    }

    // Stream ended without explicit done signal.
    yield { type: "done" };
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

interface RequestParts {
  current_content: string;
  conversation_history: Record<string, unknown>[];
  function_call_output: Record<string, unknown> | null;
  previous_response_id: string | null;
}

/** Exposed for tests. */
export function buildRequestParts(messages: Message[]): RequestParts {
  const last = messages[messages.length - 1];
  let func_output: Record<string, unknown> | null = null;

  if (Array.isArray(last.content)) {
    for (const block of last.content) {
      if (block.type === "tool_result") {
        const prev_idx = Math.max(0, messages.length - 2);
        const prev = messages[prev_idx];
        if (prev && Array.isArray(prev.content)) {
          for (const pb of prev.content) {
            if (pb.type === "tool_use" && pb.id === block.tool_use_id) {
              func_output = {
                call_id: block.tool_use_id,
                name: pb.name,
                output: block.content,
              };
              break;
            }
          }
        }
        break;
      }
    }
  }

  if (func_output !== null) {
    const assistant_idx = Math.max(0, messages.length - 2);
    const assistant = messages[assistant_idx];
    const md = assistant?.metadata as { response_id?: unknown } | undefined;
    const response_id_from_metadata =
      md && typeof md.response_id === "string" ? md.response_id : null;

    const history = messagesToHistory(messages.slice(0, Math.max(0, messages.length - 2)));
    return {
      current_content: "",
      conversation_history: history,
      function_call_output: func_output,
      previous_response_id: response_id_from_metadata,
    };
  }

  const content = last.textOrSummary();
  const history = messagesToHistory(messages.slice(0, Math.max(0, messages.length - 1)));
  return {
    current_content: content,
    conversation_history: history,
    function_call_output: null,
    previous_response_id: null,
  };
}

function messagesToHistory(messages: Message[]): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const m of messages) {
    if (m.role === "system") continue;
    const text = m.textOrSummary();
    if (m.role === "assistant" && text.trim().length === 0) continue;
    const role = m.role === "tool" ? "user" : m.role;
    out.push({ role, content: text });
  }
  return out;
}

interface SseEvent {
  eventType: string;
  data: string;
}

/** Exposed for tests. */
export async function* parseSseEvents(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let pos: number;
      while ((pos = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, pos);
        buffer = buffer.slice(pos + 2);
        let eventType: string | undefined;
        let data: string | undefined;
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) eventType = line.slice("event: ".length);
          else if (line.startsWith("data: ")) data = line.slice("data: ".length);
        }
        if (eventType && data !== undefined) {
          yield { eventType, data };
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
