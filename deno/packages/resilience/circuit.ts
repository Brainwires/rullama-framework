/**
 * Circuit-breaker decorator.
 *
 * Tracks consecutive failures per model key and opens the circuit when a
 * threshold is crossed. While open, calls fail fast. After a cooldown the
 * breaker enters half-open: the next call is a probe; success closes the
 * circuit, failure reopens it.
 *
 * Equivalent to Rust's `brainwires_resilience::circuit` module. Because the
 * Deno Provider interface doesn't expose a model name in ChatOptions, the
 * breaker defaults to a single "default" model key. A custom `modelKey`
 * function can be supplied when constructing the breaker.
 */

import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@brainwires/core";
import { ResilienceError } from "./error.ts";

/** Circuit state for a single provider/model key. */
export type CircuitState = "closed" | "open" | "half_open";

/** Circuit-breaker configuration. */
export interface CircuitBreakerConfig {
  /** Consecutive failures required to open the circuit. */
  failure_threshold: number;
  /** How long a tripped circuit stays Open before entering HalfOpen, in ms. */
  cooldown_ms: number;
}

/** Default config: 5 failures → open for 30s. */
export function defaultCircuitBreakerConfig(): CircuitBreakerConfig {
  return { failure_threshold: 5, cooldown_ms: 30_000 };
}

interface Entry {
  state: CircuitState;
  failures: number;
  open_until_ms: number | null;
}

/** Circuit-breaker decorator. */
export class CircuitBreakerProvider implements Provider {
  readonly inner: Provider;
  private readonly cfg: CircuitBreakerConfig;
  private readonly entries = new Map<string, Entry>();
  private readonly modelKey: (options: ChatOptions) => string;
  private fallback: Provider | null = null;

  constructor(
    inner: Provider,
    cfg: CircuitBreakerConfig = defaultCircuitBreakerConfig(),
    modelKey: (options: ChatOptions) => string = () => "default",
  ) {
    this.inner = inner;
    this.cfg = cfg;
    this.modelKey = modelKey;
  }

  get name(): string {
    return this.inner.name;
  }

  maxOutputTokens(): number {
    return this.inner.maxOutputTokens?.() ?? Infinity;
  }

  /** Attach a fallback provider used when the circuit is open. */
  withFallback(fallback: Provider): this {
    this.fallback = fallback;
    return this;
  }

  /** Inspect the current state for a given model key. */
  stateFor(model: string): CircuitState {
    const key = this.key(model);
    return this.entries.get(key)?.state ?? "closed";
  }

  private key(model: string): string {
    return `${this.inner.name}::${model}`;
  }

  private transitionIn(key: string): void {
    const entry = this.entries.get(key) ?? {
      state: "closed" as CircuitState,
      failures: 0,
      open_until_ms: null,
    };
    if (entry.state === "open") {
      if (entry.open_until_ms !== null && Date.now() >= entry.open_until_ms) {
        entry.state = "half_open";
        this.entries.set(key, entry);
      } else {
        throw ResilienceError.circuitOpen(this.inner.name, key, entry.failures);
      }
    } else {
      this.entries.set(key, entry);
    }
  }

  private recordSuccess(key: string): void {
    this.entries.set(key, { state: "closed", failures: 0, open_until_ms: null });
  }

  private recordFailure(key: string): void {
    const entry = this.entries.get(key) ?? {
      state: "closed" as CircuitState,
      failures: 0,
      open_until_ms: null,
    };
    entry.failures += 1;
    if (entry.failures >= this.cfg.failure_threshold || entry.state === "half_open") {
      entry.state = "open";
      entry.open_until_ms = Date.now() + this.cfg.cooldown_ms;
    }
    this.entries.set(key, entry);
  }

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    const model_label = this.modelKey(options);
    const key = this.key(model_label);
    try {
      this.transitionIn(key);
    } catch (e) {
      if (this.fallback !== null) {
        return await this.fallback.chat(messages, tools, options);
      }
      throw e;
    }
    try {
      const resp = await this.inner.chat(messages, tools, options);
      this.recordSuccess(key);
      return resp;
    } catch (e) {
      this.recordFailure(key);
      throw e;
    }
  }

  streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    // Streaming bypasses the breaker — partial streams are ambiguous.
    return this.inner.streamChat(messages, tools, options);
  }
}
