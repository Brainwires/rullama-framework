/**
 * @brainwires/agents — evaluation harness.
 *
 * Full port of Rust's `brainwires-agents::eval` subsystem:
 *
 * | Module | Purpose |
 * |---|---|
 * | {@link trial}          | Per-trial results + Wilson-score 95% CI |
 * | {@link case}           | {@link EvaluationCase} interface + built-in helpers |
 * | {@link suite}          | N-trial Monte Carlo runner |
 * | {@link recorder}       | Record + diff tool call sequences |
 * | {@link ranking_metrics}| NDCG@K, MRR, Precision@K pure helpers |
 * | {@link adversarial}    | Prompt injection, ambiguity, budget stress templates |
 * | {@link regression}     | Baseline comparison for CI gating |
 * | {@link fault_report}   | Classify suite results into priority-sorted faults |
 * | {@link fixtures}       | YAML golden-prompt fixtures |
 * | {@link stability_tests}| Long-horizon loop / goal preservation sims |
 */

// Trial + stats
export {
  type ConfidenceInterval95,
  type EvaluationStats,
  evaluationStatsFromTrials,
  percentile,
  type TrialResult,
  trialFailure,
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
  LoopDetectionSimCase,
  longHorizonStabilitySuite,
} from "./stability_tests.ts";
