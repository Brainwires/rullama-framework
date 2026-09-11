/**
 * @module @rullama/eval
 *
 * Evaluation harness for LLM agents (port of Rust's `rullama-eval` crate):
 *
 * | Module | Purpose |
 * |---|---|
 * | `trial.ts`          | Per-trial results + Wilson-score 95% CI |
 * | `case.ts`           | {@link EvaluationCase} interface + built-in helpers |
 * | `suite.ts`          | N-trial Monte Carlo runner |
 * | `recorder.ts`       | Record + diff tool call sequences |
 * | `ranking_metrics.ts`| NDCG@K, MRR, Precision@K pure helpers |
 * | `adversarial.ts`    | Prompt injection, ambiguity, budget stress templates |
 * | `regression.ts`     | Baseline comparison for CI gating |
 * | `fault_report.ts`   | Classify suite results into priority-sorted faults |
 * | `fixtures.ts`       | YAML golden-prompt fixtures |
 * | `stability_tests.ts`| Long-horizon loop / goal preservation sims |
 */

// Trial + stats
export {
  type ConfidenceInterval95,
  type EvaluationStats,
  evaluationStatsFromTrials,
  percentile,
  trialFailure,
  type TrialResult,
  trialSuccess,
  trialWithMeta,
  wilsonInterval,
} from "./trial.ts";

// Case
export {
  AlwaysFailCase,
  AlwaysPassCase,
  type EvaluationCase,
  StochasticCase,
} from "./case.ts";

// Suite
export {
  defaultSuiteConfig,
  EvaluationSuite,
  failingCases,
  overallSuccessRate,
  type SuiteConfig,
  type SuiteResult,
} from "./suite.ts";

// Recorder
export {
  computeSequenceDiff,
  isExactMatch,
  levenshtein,
  type SequenceDiff,
  type ToolCallRecord,
  ToolSequenceRecorder,
} from "./recorder.ts";

// Ranking metrics
export { mrr, ndcgAtK, precisionAtK } from "./ranking_metrics.ts";

// Adversarial
export {
  type AdversarialTestCase,
  type AdversarialTestType,
  ambiguousInstructionCase,
  budgetExhaustionCase,
  caseCategory as adversarialCaseCategory,
  categoryName as adversarialCategoryName,
  injectionPayload,
  missingContextCase,
  promptInjectionCase,
  standardAdversarialSuite,
  withExpectRejection,
} from "./adversarial.ts";

// Regression
export {
  type CategoryBaseline,
  type CategoryRegressionResult,
  defaultRegressionConfig,
  failingCategoryResults,
  improvedCategoryResults,
  isCiPassing,
  newCategoryBaseline,
  type RegressionConfig,
  type RegressionResult,
  RegressionSuite,
} from "./regression.ts";

// Fault report
export {
  analyzeSuiteForFaults,
  type FaultKind,
  faultKindLabel,
  faultKindPriority,
  type FaultReport,
  faultReportPriority,
  newCapabilityFault,
  regressionFault,
} from "./fault_report.ts";

// Fixtures
export {
  type Assertion,
  defaultRunOutcome,
  evaluate,
  type ExpectedBehavior,
  type Fixture,
  FixtureCase,
  type FixtureMessage,
  type FixtureRunner,
  loadFixtureFile,
  loadFixturesFromDir,
  type RunOutcome,
} from "./fixtures.ts";

// Stability tests
export {
  GoalPreservationCase,
  longHorizonStabilitySuite,
  LoopDetectionSimCase,
} from "./stability_tests.ts";
