/**
 * @module @rullama/mdap
 *
 * Multi-Dimensional Adaptive Planning — the MAKER voting framework for
 * reliable multi-step agent execution. Provides first-to-ahead-by-k
 * consensus voting (with RASC early stopping, confidence weighting and Borda
 * count), red-flag validation of microagent outputs, task decomposition and
 * composition helpers, cost/probability scaling laws, execution metrics and
 * structured tool-intent parsing. Equivalent to Rust's `rullama-mdap` crate.
 *
 * Building blocks:
 *
 * - **Voting**: First-to-ahead-by-k consensus algorithm for error correction
 * - **Microagents**: Minimal context single-step agents (m=1 decomposition)
 * - **Decomposition**: Task decomposition strategies (binary recursive, sequential)
 * - **Red Flags**: Output validation and format checking
 * - **Scaling**: Cost/probability estimation and optimization
 * - **Metrics**: Execution metrics collection and reporting
 * - **Composer**: Result composition from subtask outputs
 * - **Tool Intent**: Structured tool calling intent for stateless execution
 */

// Types
export {
  type CompositionFunction,
  type CompositionHandler,
  type ConfigSummary,
  type DecomposeContext,
  type DecompositionErrorDetails,
  type DecompositionResult,
  type DecompositionStrategy,
  type EarlyStoppingConfig,
  MdapError,
  type MdapErrorKind,
  type MdapEstimate,
  type MdapResult,
  type MicroagentConfig,
  type MicroagentProvider,
  type MicroagentResponse,
  type ModelCosts,
  type OutputFormat,
  type RedFlagConfig,
  type RedFlagErrorDetails,
  type RedFlagReason,
  type RedFlagResult,
  type ResponseMetadata,
  type SampledResponse,
  type Subtask,
  type SubtaskMetric,
  type SubtaskOutput,
  type SubtaskOutputWithIntent,
  type ToolCategory,
  type ToolIntent,
  type ToolSchema,
  type VoteResult,
  type VotingErrorDetails,
  type VotingMethod,
  type VotingRoundMetric,
} from "./types.ts";

// Planner: configs, helpers, scaling, metrics, composition, red-flag, tool intent
export {
  AcceptAllValidator,
  aggressiveEarlyStopping,
  atomicDecomposition,
  // Borda count
  bordaCountWinner,
  calculateExpectedCost,
  calculateExpectedVotes,
  // Scaling laws
  calculateKMin,
  calculatePFull,
  childContext,
  // Composer
  Composer,
  compositeDecomposition,
  conservativeEarlyStopping,
  // Subtask helpers
  createAtomicSubtask,
  createSubtask,
  createSubtaskOutput,
  // Decomposition helpers
  defaultDecomposeContext,
  // Early stopping presets
  defaultEarlyStopping,
  // Microagent config
  defaultMicroagentConfig,
  disabledEarlyStopping,
  estimateCallCost,
  estimateMdap,
  estimatePerStepSuccess,
  estimateValidResponseRate,
  // Confidence extraction
  extractResponseConfidence,
  // Metrics
  MdapMetrics,
  MODEL_COSTS,
  outputFormatDescription,
  // Output format helpers
  outputFormatMatches,
  parseToolIntent,
  readOnlyCategories,
  relaxedRedFlagConfig,
  sideEffectCategories,
  // Red-flag validators
  StandardRedFlagValidator,
  // Red-flag config presets
  strictRedFlagConfig,
  suggestKForBudget,
  // Tool intent helpers
  toolCategoryContains,
  toolSchemaToPrompt,
  topologicalSort,
  validateDecomposition,
} from "./planner.ts";

// Voter
export {
  FirstToAheadByKVoter,
  type RedFlagValidator,
  VoterBuilder,
} from "./voter.ts";
