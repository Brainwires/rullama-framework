import type { ChatResponse, Message, StreamChunk } from "./message.ts";
import type { Tool } from "./tool.ts";

/** Base provider interface for AI providers.
 * Equivalent to Rust's `Provider` trait in rullama-core. */
export interface Provider {
  /** Get the provider name. */
  readonly name: string;

  /** Get the model's maximum output tokens. Returns undefined if no specific limit. */
  maxOutputTokens?(): number | undefined;

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
  /** Sampling temperature (default 0.7). */
  temperature?: number;
  /** Maximum tokens the model may generate (default 4096). */
  max_tokens?: number;
  /** Nucleus-sampling probability mass. */
  top_p?: number;
  /** Sequences that end generation when produced. */
  stop?: string[];
  /** System prompt sent ahead of the messages. */
  system?: string;
  /**
   * Model to use for this call, when the provider serves several. Providers
   * bound to one model ignore it; decorators (circuit breaker, cache) key
   * their state per model with it.
   */
  model?: string;

  /** Create options, applying the defaults (`temperature` 0.7, `max_tokens` 4096) for anything not given. */
  constructor(opts?: Partial<ChatOptions>) {
    this.temperature = opts?.temperature ?? 0.7;
    this.max_tokens = opts?.max_tokens ?? 4096;
    this.top_p = opts?.top_p;
    this.stop = opts?.stop;
    this.system = opts?.system;
    this.model = opts?.model;
  }

  /** Set the model (builder). */
  setModel(model: string): this {
    this.model = model;
    return this;
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
    if (this.model !== undefined) obj.model = this.model;
    return obj;
  }
}
