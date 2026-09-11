/**
 * Configuration for local inference components.
 *
 * Equivalent to Rust's `rullama_reasoning::LocalInferenceConfig`.
 */

/**
 * Feature flags and per-task model ids for the local-inference scorers.
 * Tier 1 covers routing, validation and complexity; Tier 2 covers the
 * context/retrieval tasks (of which only retrieval gating is ported so far —
 * the other Tier 2 flags are carried for parity with the Rust crate).
 */
export interface LocalInferenceConfig {
  // TIER 1 — quick wins
  /** Run {@link LocalRouter} to pick tool categories for a query. */
  routing_enabled: boolean;
  /** Run {@link LocalValidator} on agent responses. */
  validation_enabled: boolean;
  /** Run {@link ComplexityScorer} on task descriptions. */
  complexity_enabled: boolean;
  // TIER 2 — context & retrieval
  /** Summarize long context (Rust-only today; no Deno summarizer yet). */
  summarization_enabled: boolean;
  /** Run {@link RetrievalClassifier} before fetching earlier context. */
  retrieval_gating_enabled: boolean;
  /** Score retrieved chunks for relevance (Rust-only today). */
  relevance_scoring_enabled: boolean;
  /** Pick a reasoning strategy per task (Rust-only today). */
  strategy_selection_enabled: boolean;
  /** Enhance entity extraction with a local model (Rust-only today). */
  entity_enhancement_enabled: boolean;

  // Per-task model selection
  /** Model id for routing; `null` means use the caller's default. */
  routing_model: string | null;
  /** Model id for validation; `null` means use the caller's default. */
  validation_model: string | null;
  /** Model id for complexity scoring; `null` means use the caller's default. */
  complexity_model: string | null;
  /** Model id for summarization; `null` means use the caller's default. */
  summarization_model: string | null;
  /** Model id for retrieval gating; `null` means use the caller's default. */
  retrieval_model: string | null;
  /** Model id for relevance scoring; `null` means use the caller's default. */
  relevance_model: string | null;
  /** Model id for strategy selection; `null` means use the caller's default. */
  strategy_model: string | null;
  /** Model id for entity enhancement; `null` means use the caller's default. */
  entity_model: string | null;

  /** Emit a `local_llm task=… latency_ms=…` line per inference (see {@link InferenceTimer}). */
  log_inference: boolean;
}

function base(): LocalInferenceConfig {
  return {
    routing_enabled: false,
    validation_enabled: false,
    complexity_enabled: false,
    summarization_enabled: false,
    retrieval_gating_enabled: false,
    relevance_scoring_enabled: false,
    strategy_selection_enabled: false,
    entity_enhancement_enabled: false,
    routing_model: "lfm2-350m",
    validation_model: "lfm2-350m",
    complexity_model: "lfm2-350m",
    summarization_model: "lfm2-1.2b",
    retrieval_model: "lfm2-350m",
    relevance_model: "lfm2-350m",
    strategy_model: "lfm2-1.2b",
    entity_model: "lfm2-350m",
    log_inference: true,
  };
}

/**
 * All scorers disabled, `log_inference` on, and the advisory default model
 * ids (`lfm2-350m` for the small tasks, `lfm2-1.2b` for summarization and
 * strategy selection).
 */
export function defaultLocalInferenceConfig(): LocalInferenceConfig {
  return base();
}

/** Defaults with the Tier 1 flags (routing, validation, complexity) on. */
export function tier1Enabled(): LocalInferenceConfig {
  return {
    ...base(),
    routing_enabled: true,
    validation_enabled: true,
    complexity_enabled: true,
  };
}

/**
 * Defaults with only the Tier 2 flags (summarization, retrieval gating,
 * relevance scoring, strategy selection, entity enhancement) on.
 */
export function tier2Enabled(): LocalInferenceConfig {
  return {
    ...base(),
    summarization_enabled: true,
    retrieval_gating_enabled: true,
    relevance_scoring_enabled: true,
    strategy_selection_enabled: true,
    entity_enhancement_enabled: true,
  };
}

/** Defaults with every Tier 1 and Tier 2 flag on. */
export function allEnabled(): LocalInferenceConfig {
  return {
    ...base(),
    routing_enabled: true,
    validation_enabled: true,
    complexity_enabled: true,
    summarization_enabled: true,
    retrieval_gating_enabled: true,
    relevance_scoring_enabled: true,
    strategy_selection_enabled: true,
    entity_enhancement_enabled: true,
  };
}

/** Measure inference latency and optionally log the outcome. */
export class InferenceTimer {
  private readonly start = performance.now();
  /** Task label written to the log line (e.g. `"routing"`). */
  readonly task: string;
  /** Model id written to the log line. */
  readonly model: string;

  /**
   * Start the clock immediately (the timer captures `performance.now()` at
   * construction).
   *
   * @param task Task label for the log line.
   * @param model Model id for the log line.
   */
  constructor(task: string, model: string) {
    this.task = task;
    this.model = model;
  }

  /** Milliseconds elapsed since construction. */
  elapsedMs(): number {
    return performance.now() - this.start;
  }

  /**
   * Stop timing and optionally log
   * `local_llm task=<task> model=<model> latency_ms=<n> ok|fallback`.
   *
   * @param success `true` if the local model produced a usable answer,
   *   `false` if the caller fell back to a heuristic.
   * @param log Sink for the log line; pass `null` to skip logging.
   * @returns The elapsed milliseconds.
   */
  finish(success: boolean, log: ((msg: string) => void) | null = null): number {
    const ms = this.elapsedMs();
    if (log) {
      log(
        `local_llm task=${this.task} model=${this.model} latency_ms=${
          ms.toFixed(0)
        } ${success ? "ok" : "fallback"}`,
      );
    }
    return ms;
  }
}
