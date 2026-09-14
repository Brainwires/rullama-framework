/**
 * Budget decorator — caps on tokens, USD, and rounds.
 *
 * Deno is single-threaded per isolate, so the Rust atomic counters are just
 * plain numbers here. The guard is still cheap to share by reference.
 *
 * Equivalent to Rust's `rullama_resilience::budget` module.
 */

import { ProviderDecorator } from "./decorator.ts";
import type {
  ChatOptions,
  ChatResponse,
  ContentBlock,
  Message,
  Provider,
  StreamChunk,
  Tool,
  Usage,
} from "@rullama/core";
import { ResilienceError } from "./error.ts";

/** Caps to enforce on a single BudgetGuard. null = unbounded. */
export interface BudgetConfig {
  /** Cap on cumulative `usage.total_tokens` observed across calls (input estimate is also pre-checked). */
  max_tokens: number | null;
  /** Cap on cumulative spend recorded via {@link BudgetGuard.recordCostCents}, in USD cents. */
  max_usd_cents: number | null;
  /** Cap on the number of rounds (one per `chat`/`streamChat` call or `checkAndTick`). */
  max_rounds: number | null;
}

/** Default BudgetConfig with no caps. */
export function defaultBudgetConfig(): BudgetConfig {
  return { max_tokens: null, max_usd_cents: null, max_rounds: null };
}

/** Shared mutable budget counters. */
export class BudgetGuard {
  /** The caps this guard enforces. */
  readonly cfg: BudgetConfig;
  /** Running total of tokens recorded through {@link recordUsage}. */
  private tokens = 0;
  /** Running total of spend recorded through {@link recordCostCents}. */
  private usd_cents = 0;
  /** Number of rounds ticked so far. */
  private rounds = 0;

  /**
   * Create a guard with all counters at zero.
   *
   * @param cfg Caps to enforce; defaults to unbounded ({@link defaultBudgetConfig}).
   */
  constructor(cfg: BudgetConfig = defaultBudgetConfig()) {
    this.cfg = cfg;
  }

  /** The caps this guard was constructed with (same object as {@link cfg}). */
  config(): BudgetConfig {
    return this.cfg;
  }

  /** Tokens consumed so far, as accumulated from provider `usage` reports. */
  tokensConsumed(): number {
    return this.tokens;
  }

  /** Spend consumed so far in USD cents, as recorded by {@link recordCostCents}. */
  usdCentsConsumed(): number {
    return this.usd_cents;
  }

  /** Rounds consumed so far — one per decorated call or explicit tick. */
  roundsConsumed(): number {
    return this.rounds;
  }

  /** Zero every counter; the caps are unchanged. */
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
      throw ResilienceError.budgetExceeded(
        "tokens",
        this.tokens,
        this.cfg.max_tokens,
      );
    }
    if (
      this.cfg.max_usd_cents !== null &&
      this.usd_cents >= this.cfg.max_usd_cents
    ) {
      throw ResilienceError.budgetExceeded(
        "usd_cents",
        this.usd_cents,
        this.cfg.max_usd_cents,
      );
    }
    if (this.cfg.max_rounds !== null && this.rounds >= this.cfg.max_rounds) {
      throw ResilienceError.budgetExceeded(
        "rounds",
        this.rounds,
        this.cfg.max_rounds,
      );
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
export class BudgetProvider extends ProviderDecorator {
  /** The shared counters/caps consulted before and after every call. */
  readonly guard: BudgetGuard;

  /**
   * Wrap `inner` so every call is checked against and recorded into `guard`.
   * Share one guard between several providers to enforce a joint budget.
   */
  constructor(inner: Provider, guard: BudgetGuard) {
    super(inner);
    this.guard = guard;
  }

  /**
   * Run the pre-flight checks, tick a round, forward the call, then record
   * the response's `usage` into the guard.
   *
   * Throws a `budget_exceeded` {@link ResilienceError} when a cap is already
   * hit or when the estimated input tokens alone would exceed `max_tokens`.
   */
  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    this.guard.check();

    // Pre-flight: reject if the raw payload alone would blow the token cap.
    if (this.guard.cfg.max_tokens !== null) {
      const projected = this.guard.tokensConsumed() +
        approxInputTokens(messages);
      if (projected > this.guard.cfg.max_tokens) {
        throw ResilienceError.budgetExceeded(
          "tokens",
          projected,
          this.guard.cfg.max_tokens,
        );
      }
    }

    this.guard.tickRounds();
    const resp = await this.inner.chat(messages, tools, options);
    this.guard.recordUsage(resp.usage);
    return resp;
  }

  /**
   * Same pre-flight checks and round tick as {@link chat}, performed
   * synchronously before the stream starts; `usage` chunks observed while
   * iterating are recorded into the guard.
   */
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
        throw ResilienceError.budgetExceeded(
          "tokens",
          projected,
          guard.cfg.max_tokens,
        );
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
