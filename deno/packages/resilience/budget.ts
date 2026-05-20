/**
 * Budget decorator — caps on tokens, USD, and rounds.
 *
 * Deno is single-threaded per isolate, so the Rust atomic counters are just
 * plain numbers here. The guard is still cheap to share by reference.
 *
 * Equivalent to Rust's `brainwires_resilience::budget` module.
 */

import type {
  ChatOptions,
  ChatResponse,
  ContentBlock,
  Message,
  Provider,
  StreamChunk,
  Tool,
  Usage,
} from "@brainwires/core";
import { ResilienceError } from "./error.ts";

/** Caps to enforce on a single BudgetGuard. null = unbounded. */
export interface BudgetConfig {
  max_tokens: number | null;
  max_usd_cents: number | null;
  max_rounds: number | null;
}

/** Default BudgetConfig with no caps. */
export function defaultBudgetConfig(): BudgetConfig {
  return { max_tokens: null, max_usd_cents: null, max_rounds: null };
}

/** Shared mutable budget counters. */
export class BudgetGuard {
  readonly cfg: BudgetConfig;
  private tokens = 0;
  private usd_cents = 0;
  private rounds = 0;

  constructor(cfg: BudgetConfig = defaultBudgetConfig()) {
    this.cfg = cfg;
  }

  config(): BudgetConfig {
    return this.cfg;
  }

  tokensConsumed(): number {
    return this.tokens;
  }

  usdCentsConsumed(): number {
    return this.usd_cents;
  }

  roundsConsumed(): number {
    return this.rounds;
  }

  reset(): void {
    this.tokens = 0;
    this.usd_cents = 0;
    this.rounds = 0;
  }

  /**
   * Pre-flight check. Throws {@link ResilienceError} (kind "budget_exceeded")
   * if any cap has already been reached.
   */
  check(): void {
    if (this.cfg.max_tokens !== null && this.tokens >= this.cfg.max_tokens) {
      throw ResilienceError.budgetExceeded("tokens", this.tokens, this.cfg.max_tokens);
    }
    if (this.cfg.max_usd_cents !== null && this.usd_cents >= this.cfg.max_usd_cents) {
      throw ResilienceError.budgetExceeded("usd_cents", this.usd_cents, this.cfg.max_usd_cents);
    }
    if (this.cfg.max_rounds !== null && this.rounds >= this.cfg.max_rounds) {
      throw ResilienceError.budgetExceeded("rounds", this.rounds, this.cfg.max_rounds);
    }
  }

  /** Check caps then tick the rounds counter — call once per agent iteration. */
  checkAndTick(): void {
    this.check();
    this.rounds += 1;
  }

  /** Accumulate observed usage into the counters. */
  recordUsage(usage: Usage): void {
    this.tokens += usage.total_tokens;
  }

  /** Accumulate observed spend (USD cents). */
  recordCostCents(cents: number): void {
    this.usd_cents += cents;
  }

  /** Internal — increment the rounds counter without checking. */
  tickRounds(): void {
    this.rounds += 1;
  }
}

/** Rough character-level token estimate. Exposed for tests. */
export function approxInputTokens(messages: Message[]): number {
  let chars = 0;
  for (const m of messages) {
    if (typeof m.content === "string") {
      chars += m.content.length;
    } else {
      for (const b of m.content) chars += approxBlockLen(b);
    }
  }
  // ~4 chars per token (BPE heuristic).
  return Math.floor(chars / 4);
}

function approxBlockLen(b: ContentBlock): number {
  switch (b.type) {
    case "text":
      return b.text.length;
    case "tool_use":
      return JSON.stringify(b.input).length;
    case "tool_result":
      return b.content.length;
    case "image":
      return 512;
  }
}

/** A Provider decorator that enforces a {@link BudgetGuard} around every call. */
export class BudgetProvider implements Provider {
  readonly inner: Provider;
  readonly guard: BudgetGuard;

  constructor(inner: Provider, guard: BudgetGuard) {
    this.inner = inner;
    this.guard = guard;
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
    this.guard.check();

    // Pre-flight: reject if the raw payload alone would blow the token cap.
    if (this.guard.cfg.max_tokens !== null) {
      const projected = this.guard.tokensConsumed() + approxInputTokens(messages);
      if (projected > this.guard.cfg.max_tokens) {
        throw ResilienceError.budgetExceeded("tokens", projected, this.guard.cfg.max_tokens);
      }
    }

    this.guard.tickRounds();
    const resp = await this.inner.chat(messages, tools, options);
    this.guard.recordUsage(resp.usage);
    return resp;
  }

  streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const guard = this.guard;
    const upstream = () => this.inner.streamChat(messages, tools, options);

    // Fail-fast check synchronously so the iterator yields an error promptly.
    guard.check();
    if (guard.cfg.max_tokens !== null) {
      const projected = guard.tokensConsumed() + approxInputTokens(messages);
      if (projected > guard.cfg.max_tokens) {
        throw ResilienceError.budgetExceeded("tokens", projected, guard.cfg.max_tokens);
      }
    }
    guard.tickRounds();

    return (async function* () {
      for await (const chunk of upstream()) {
        if (chunk.type === "usage") {
          guard.recordUsage(chunk.usage);
        }
        yield chunk;
      }
    })();
  }
}
