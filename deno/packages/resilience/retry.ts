/**
 * Retry decorator with exponential backoff + jitter.
 *
 * Equivalent to Rust's `brainwires_resilience::retry` module.
 */

import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@brainwires/core";
import { classifyError, isRetryable, parseRetryAfter } from "./classify.ts";
import { ResilienceError } from "./error.ts";

/** Configuration for retry behavior. */
export interface RetryPolicy {
  /** Maximum number of attempts including the first. 1 disables retry. */
  max_attempts: number;
  /** Base backoff in ms. Effective delay is `base * 2^(attempt-1)` clamped to `max`. */
  base_ms: number;
  /** Upper bound on a single sleep, in ms. */
  max_ms: number;
  /** Proportional jitter applied to each delay (0.0..=1.0). */
  jitter: number;
  /** If true, honor `retry-after` hints in error messages. */
  honor_retry_after: boolean;
  /** Hard wall-clock ceiling for the retry sequence, in ms. null disables it. */
  overall_deadline_ms: number | null;
}

/** Sensible default policy. */
export function defaultRetryPolicy(): RetryPolicy {
  return {
    max_attempts: 4,
    base_ms: 500,
    max_ms: 30_000,
    jitter: 0.2,
    honor_retry_after: true,
    overall_deadline_ms: 60_000,
  };
}

/** Disable retries entirely. */
export function noRetryPolicy(): RetryPolicy {
  return { ...defaultRetryPolicy(), max_attempts: 1 };
}

/** Compute exponential backoff for `attempt` (1-indexed). Exposed for tests. */
export function backoffFor(policy: RetryPolicy, attempt: number): number {
  const shift = Math.min(Math.max(0, attempt - 1), 16);
  const nominal = policy.base_ms * (1 << shift);
  const capped = Math.min(nominal, policy.max_ms);
  return applyJitter(capped, policy.jitter);
}

/** Apply proportional jitter. Exposed for tests. */
export function applyJitter(base_ms: number, factor: number): number {
  if (factor <= 0) return base_ms;
  const clamped = Math.min(1, Math.max(0, factor));
  const spread = base_ms * clamped;
  const delta = (Math.random() * 2 - 1) * spread;
  return Math.max(0, Math.round(base_ms + delta));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * A Provider decorator that retries transient failures with exponential
 * backoff and optional jitter. Only non-streaming `chat` requests are
 * retried — streaming passes through unchanged.
 */
export class RetryProvider implements Provider {
  readonly inner: Provider;
  private readonly policy: RetryPolicy;

  constructor(inner: Provider, policy: RetryPolicy = defaultRetryPolicy()) {
    this.inner = inner;
    this.policy = policy;
  }

  get name(): string {
    return this.inner.name;
  }

  maxOutputTokens(): number {
    return this.inner.maxOutputTokens?.() ?? Infinity;
  }

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    let last_err: Error | undefined;
    const started = Date.now();

    for (let attempt = 1; attempt <= this.policy.max_attempts; attempt++) {
      try {
        return await this.inner.chat(messages, tools, options);
      } catch (raw) {
        const err = raw instanceof Error ? raw : new Error(String(raw));
        const cls = classifyError(err);
        if (!isRetryable(cls) || attempt === this.policy.max_attempts) {
          if (attempt > 1) {
            throw ResilienceError.retriesExhausted(attempt, err);
          }
          throw err;
        }

        let delay = this.policy.honor_retry_after
          ? (parseRetryAfter(err) ?? backoffFor(this.policy, attempt))
          : backoffFor(this.policy, attempt);

        if (this.policy.overall_deadline_ms !== null) {
          const elapsed = Date.now() - started;
          const deadline = this.policy.overall_deadline_ms;
          if (elapsed >= deadline || elapsed + delay >= deadline) {
            throw ResilienceError.deadlineExceeded(attempt, elapsed, err);
          }
          const remaining = deadline - elapsed;
          if (delay > remaining) delay = remaining;
        }

        last_err = err;
        await sleep(delay);
      }
    }

    throw ResilienceError.retriesExhausted(
      this.policy.max_attempts,
      last_err ?? new Error("unknown retry failure"),
    );
  }

  streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    // Streaming responses aren't retried — see module docs.
    return this.inner.streamChat(messages, tools, options);
  }
}
