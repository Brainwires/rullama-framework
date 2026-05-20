/**
 * @module @brainwires/reasoning
 *
 * Layer 3 — Intelligence. Provider-agnostic reasoning primitives for the
 * Brainwires Agent Framework.
 *
 * ## What's here
 *
 * - **Parsers** — `OutputParser` (re-exported from `@brainwires/core`) and
 *   {@link parsePlanSteps} / {@link stepsToTasks}.
 * - **Local scorers** — `ComplexityScorer`, `LocalRouter`, `LocalValidator`,
 *   `RetrievalClassifier`. Each takes a `Provider` and falls back to a
 *   pattern-based heuristic when the LLM call fails.
 * - **Config + timer** — `LocalInferenceConfig`, `InferenceTimer`.
 *
 * ## Intentionally deferred
 *
 * The Rust crate additionally ships `strategies`, `strategy_selector`,
 * `summarizer`, `relevance_scorer`, and `entity_enhancer`. These are
 * scheduled for a follow-up port — the current Deno slice covers the
 * Tier-1 fast path (routing, validation, complexity, retrieval gating)
 * plus the parsers the rest of the framework consumes.
 *
 * Equivalent to Rust's `brainwires-reasoning` crate.
 */

// Parsers — OutputParser already lives in @brainwires/core (see plan §B1).
export {
  extractJson,
  JsonListParser,
  JsonOutputParser,
  type OutputParser,
  RegexOutputParser,
} from "@brainwires/core";

export { type ParsedStep, parsePlanSteps, stepsToTasks } from "./plan_parser.ts";

// Config + timer
export {
  allEnabled,
  defaultLocalInferenceConfig,
  InferenceTimer,
  type LocalInferenceConfig,
  tier1Enabled,
  tier2Enabled,
} from "./config.ts";

// Complexity scorer
export {
  type ComplexityResult,
  ComplexityScorer,
  complexityFromLocal,
  defaultComplexity,
  parseScore as parseComplexityScore,
  scoreHeuristic,
} from "./complexity.ts";

// Router
export {
  LocalRouter,
  parseCategories,
  type RouteResult,
  routeFromFallback,
  routeFromLocal,
  type ToolCategory,
} from "./router.ts";

// Validator
export {
  isInvalid,
  isValid,
  LocalValidator,
  parseValidation,
  validateHeuristic,
  type ValidationResult,
} from "./validator.ts";

// Retrieval classifier
export {
  type ClassificationResult,
  classificationFromFallback,
  classificationFromLocal,
  classifyHeuristic as classifyRetrievalHeuristic,
  parseClassification as parseRetrievalClassification,
  RetrievalClassifier,
  type RetrievalNeed,
  retrievalScore,
  shouldRetrieve,
} from "./retrieval_classifier.ts";
