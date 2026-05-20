/**
 * Configuration for local inference components.
 *
 * Equivalent to Rust's `brainwires_reasoning::LocalInferenceConfig`.
 */

export interface LocalInferenceConfig {
  // TIER 1 — quick wins
  routing_enabled: boolean;
  validation_enabled: boolean;
  complexity_enabled: boolean;
  // TIER 2 — context & retrieval
  summarization_enabled: boolean;
  retrieval_gating_enabled: boolean;
  relevance_scoring_enabled: boolean;
  strategy_selection_enabled: boolean;
  entity_enhancement_enabled: boolean;

  // Per-task model selection
  routing_model: string | null;
  validation_model: string | null;
  complexity_model: string | null;
  summarization_model: string | null;
  retrieval_model: string | null;
  relevance_model: string | null;
  strategy_model: string | null;
  entity_model: string | null;

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

export function defaultLocalInferenceConfig(): LocalInferenceConfig {
  return base();
}

export function tier1Enabled(): LocalInferenceConfig {
  return {
    ...base(),
    routing_enabled: true,
    validation_enabled: true,
    complexity_enabled: true,
  };
}

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
  readonly task: string;
  readonly model: string;

  constructor(task: string, model: string) {
    this.task = task;
    this.model = model;
  }

  elapsedMs(): number {
    return performance.now() - this.start;
  }

  finish(success: boolean, log: ((msg: string) => void) | null = null): number {
    const ms = this.elapsedMs();
    if (log) {
      log(
        `local_llm task=${this.task} model=${this.model} latency_ms=${ms.toFixed(0)} ${success ? "ok" : "fallback"}`,
      );
    }
    return ms;
  }
}
