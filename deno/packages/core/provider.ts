import type { ChatResponse, Message, StreamChunk } from "./message.ts";
import type { Tool } from "./tool.ts";

/** Base provider interface for AI providers.
 * Equivalent to Rust's `Provider` trait in rullama-core. */
export interface Provider {
  /** Get the provider name. */
  readonly name: string;

  /** Get the model's maximum output tokens. Returns undefined if no specific limit. */
  maxOutputTokens?(): number;

  /** Chat completion (non-streaming). */
  chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse>;

  /** Chat completion (streaming). */
  streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk>;
}

/** Chat completion options.
 * Equivalent to Rust's `ChatOptions` in rullama-core. */
export class ChatOptions {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stop?: string[];
  system?: string;

  constructor(opts?: Partial<ChatOptions>) {
    this.temperature = opts?.temperature ?? 0.7;
    this.max_tokens = opts?.max_tokens ?? 4096;
    this.top_p = opts?.top_p;
    this.stop = opts?.stop;
    this.system = opts?.system;
  }

  /** Create new chat options with defaults. */
  static create(): ChatOptions {
    return new ChatOptions();
  }

  /** Set temperature (builder). */
  setTemperature(temperature: number): this {
    this.temperature = temperature;
    return this;
  }

  /** Set max tokens (builder). */
  setMaxTokens(maxTokens: number): this {
    this.max_tokens = maxTokens;
    return this;
  }

  /** Set system prompt (builder). */
  setSystem(system: string): this {
    this.system = system;
    return this;
  }

  /** Set top-p sampling (builder). */
  setTopP(topP: number): this {
    this.top_p = topP;
    return this;
  }

  /** Deterministic classification/routing (temp=0, few tokens). */
  static deterministic(maxTokens: number): ChatOptions {
    return new ChatOptions({ temperature: 0.0, max_tokens: maxTokens });
  }

  /** Low-temperature factual generation. */
  static factual(maxTokens: number): ChatOptions {
    return new ChatOptions({
      temperature: 0.1,
      max_tokens: maxTokens,
      top_p: 0.9,
    });
  }

  /** Creative generation with moderate temperature. */
  static creative(maxTokens: number): ChatOptions {
    return new ChatOptions({ temperature: 0.3, max_tokens: maxTokens });
  }

  /** Serialize to JSON, omitting undefined fields. */
  toJSON(): Record<string, unknown> {
    const obj: Record<string, unknown> = {};
    if (this.temperature !== undefined) obj.temperature = this.temperature;
    if (this.max_tokens !== undefined) obj.max_tokens = this.max_tokens;
    if (this.top_p !== undefined) obj.top_p = this.top_p;
    if (this.stop !== undefined) obj.stop = this.stop;
    if (this.system !== undefined) obj.system = this.system;
    return obj;
  }
}
