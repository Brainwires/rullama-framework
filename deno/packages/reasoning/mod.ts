/**
 * Provider-agnostic reasoning primitives for rullama (Layer 3 —
 * Intelligence). Provides the Tier-1 local-inference scorers —
 * {@link ComplexityScorer}, {@link LocalRouter}, {@link LocalValidator} and
 * {@link RetrievalClassifier} — each of which calls a `@rullama/core`
 * `Provider` and ships a pure heuristic fallback, plus the plan parser
 * ({@link parsePlanSteps} / {@link stepsToTasks}), the
 * {@link LocalInferenceConfig} feature flags with {@link InferenceTimer}, and
 * the `OutputParser` family re-exported from `@rullama/core`. The Rust crate's
 * `strategies`, `strategy_selector`, `summarizer`, `relevance_scorer` and
 * `entity_enhancer` modules are not ported yet. Equivalent to Rust's
 * `rullama-reasoning` crate.
 *
 * @module
 */

// Core types that appear in this package's public signatures.
export type { Provider, Task } from "@rullama/core";

// Parsers — OutputParser already lives in @rullama/core (see plan §B1).
export {
  extractJson,
  JsonListParser,
  JsonOutputParser,
  type OutputParser,
  RegexOutputParser,
} from "@rullama/core";

export {
  type ParsedStep,
  parsePlanSteps,
  stepsToTasks,
} from "./plan_parser.ts";

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
  complexityFromLocal,
  type ComplexityResult,
  ComplexityScorer,
  defaultComplexity,
  parseScore as parseComplexityScore,
  scoreHeuristic,
} from "./complexity.ts";

// Router
export {
  LocalRouter,
  parseCategories,
  routeFromFallback,
  routeFromLocal,
  type RouteResult,
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
  classificationFromFallback,
  classificationFromLocal,
  type ClassificationResult,
  classifyHeuristic as classifyRetrievalHeuristic,
  parseClassification as parseRetrievalClassification,
  RetrievalClassifier,
  type RetrievalNeed,
  retrievalScore,
  shouldRetrieve,
} from "./retrieval_classifier.ts";
