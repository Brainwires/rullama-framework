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
  /** The temperature the constructor fills in when none is given. */
  static readonly DEFAULT_TEMPERATURE = 0.7;

  /**
   * Sampling temperature (default 0.7). Providers whose models refuse
   * sampling parameters (current Anthropic models) send it only when the
   * caller set it; see {@link ChatOptions.explicitTemperature}.
   */
  temperature?: number;
  /**
   * True when `temperature` holds the constructor's default rather than a
   * value the caller gave. {@link ChatOptions.setTemperature} clears it, and
   * any other value assigned to `temperature` counts as explicit as well.
   * Never serialized.
   */
  temperature_defaulted?: boolean;
  /** Maximum tokens the model may generate (default 4096). */
  max_tokens?: number;
  /** Nucleus-sampling probability mass. No default. */
  top_p?: number;
  /**
   * Top-k sampling: sample only from the k most likely tokens. No default;
   * providers that support it send it only when set.
   */
  top_k?: number;
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
  /**
   * Cancels the request when aborted. Providers that honour it pass it to
   * `fetch`, so aborting mid-stream closes the connection (which stops
   * server-side generation for e.g. Ollama). Never serialized.
   */
  signal?: AbortSignal;
  /**
   * Extra HTTP headers for this call (e.g. a vendor beta flag), for the
   * providers that honour it (`AnthropicChatProvider`). They are merged over the
   * provider's own extra headers, but the auth and version headers a provider
   * sets always win. Never serialized.
   */
  extra_headers?: Record<string, string>;
  /**
   * Extra top-level request-body fields for this call, for the providers that
   * honour it (`AnthropicChatProvider`). They are merged over the provider's
   * own extra body fields. The fields the provider builds (model, messages,
   * system, max_tokens, sampling, tools, stream, …) always win. Never
   * serialized.
   */
  extra_body?: Record<string, unknown>;

  /**
   * Create options, applying the defaults (`temperature` 0.7, marked
   * `temperature_defaulted`; `max_tokens` 4096) for anything not given.
   */
  constructor(opts: Partial<ChatOptions> = {}) {
    const {
      temperature,
      max_tokens,
      top_p,
      top_k,
      stop,
      system,
      model,
      signal,
      extra_headers,
      extra_body,
    } = opts;
    if (extra_headers !== undefined) this.extra_headers = extra_headers;
    if (extra_body !== undefined) this.extra_body = extra_body;
    this.temperature = temperature ?? ChatOptions.DEFAULT_TEMPERATURE;
    // Copying defaulted options (`new ChatOptions(other)`) keeps them defaulted.
    if (ChatOptions.explicitTemperature(opts) === undefined) {
      this.temperature_defaulted = true;
    }
    this.max_tokens = max_tokens ?? 4096;
    this.top_p = top_p;
    if (top_k !== undefined) this.top_k = top_k;
    this.stop = stop;
    this.system = system;
    this.model = model;
    this.signal = signal;
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
    this.temperature_defaulted = false;
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

  /** Set top-k sampling (builder). */
  setTopK(topK: number): this {
    this.top_k = topK;
    return this;
  }

  /**
   * The temperature the caller asked for, or `undefined` when none was given:
   * the constructor's default (`temperature_defaulted` and still 0.7) does
   * not count. Plain option objects work too; any `temperature` they carry is
   * explicit unless they are marked `temperature_defaulted`.
   */
  static explicitTemperature(
    options: Pick<ChatOptions, "temperature" | "temperature_defaulted">,
  ): number | undefined {
    const t = options.temperature;
    if (t === undefined) return undefined;
    if (
      options.temperature_defaulted === true &&
      t === ChatOptions.DEFAULT_TEMPERATURE
    ) return undefined;
    return t;
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
    if (this.top_k !== undefined) obj.top_k = this.top_k;
    if (this.stop !== undefined) obj.stop = this.stop;
    if (this.system !== undefined) obj.system = this.system;
    if (this.model !== undefined) obj.model = this.model;
    return obj;
  }
}
