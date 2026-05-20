/**
 * @module @brainwires/agents
 *
 * Agent orchestration, coordination, and lifecycle management for the
 * Brainwires Agent Framework. Equivalent to Rust's `brainwires-agents` crate.
 *
 * ## Core Components
 * - **AgentRuntime / runAgentLoop** - Generic execution loop for autonomous agents
 * - **TaskAgent** - Concrete agent implementation with provider + tool loop
 * - **AgentContext** - Environment bundle (tools, hub, locks, working set)
 * - **CommunicationHub** - Inter-agent messaging bus
 * - **FileLockManager** - File access coordination with deadlock detection
 * - **TaskManager** - Hierarchical task decomposition and dependency tracking
 * - **TaskQueue** - Priority-based task scheduling
 * - **ValidationLoop** - Quality checks before agent completion
 * - **PlanExecutorAgent** - Plan execution orchestration
 *
 * ## Coordination Patterns
 * - **ContractNet** - Bidding protocol for agent negotiation
 * - **Saga** - Compensating transactions for distributed operations
 * - **OptimisticConcurrency** - Optimistic locking with conflict detection
 * - **MarketAllocator** - Market-based task allocation with bidding/auction
 * - **ThreeStateModel** - State snapshots and rollback
 * - **WaitQueue** - Queue-based synchronization primitives
 *
 * ## Specialized Agents
 * - **JudgeAgent** - LLM-powered cycle evaluator
 * - **PlannerAgent** - LLM-powered dynamic task planner
 * - **ValidatorAgent** - Standalone read-only validation agent
 * - **CycleOrchestrator** - Plan->Work->Judge loop
 *
 * ## Execution Tracking
 * - **ExecutionGraph** - DAG-based execution tracking
 * - **AgentPool** - Agent instance pooling and reuse
 *
 * ## Lifecycle Hooks
 * - **AgentLifecycleHooks** - Granular control over the agent execution loop
 *
 * ## MDAP (MAKER voting framework)
 * - **FirstToAheadByKVoter** - Consensus voting with early stopping and confidence weighting
 * - **StandardRedFlagValidator** - Output validation and format checking
 * - **Composer** - Result composition from subtask outputs
 * - **MdapMetrics** - Execution metrics collection and reporting
 * - **Scaling laws** - Cost/probability estimation (MAKER paper equations 13-18)
 * - **Tool Intent** - Structured tool calling intent for stateless execution
 */

// Runtime
export {
  runAgentLoop,
  type AgentExecutionResult,
  type AgentRuntime,
} from "./runtime.ts";

// Context
export { AgentContext, type ToolPreHook } from "./context.ts";

// Task agent
export {
  defaultLoopDetectionConfig,
  defaultTaskAgentConfig,
  formatTaskAgentStatus,
  spawnTaskAgent,
  TaskAgent,
  type FailureCategory,
  type LoopDetectionConfig,
  type TaskAgentConfig,
  type TaskAgentResult,
  type TaskAgentStatus,
} from "./task_agent.ts";

// Communication
export {
  CommunicationHub,
  type AgentMessage,
  type ConflictInfo,
  type ConflictType,
  type GitOperationType,
  type MessageEnvelope,
  type OperationType,
} from "./communication.ts";

// File locks
export {
  FileLockManager,
  isLockExpired,
  lockTimeRemaining,
  type LockGuard,
  type LockInfo,
  type LockStats,
  type LockType,
} from "./file_locks.ts";

// Task manager
export {
  formatDurationSecs,
  TaskManager,
  type TaskStats,
  type TimeStats,
} from "./task_manager.ts";

// Task queue
export { TaskQueue, type QueuedTask } from "./task_queue.ts";

// Validation loop
export {
  defaultValidationConfig,
  disabledValidationConfig,
  formatValidationFeedback,
  runValidation,
  type ValidationCheck,
  type ValidationConfig,
  type ValidationIssue,
  type ValidationResult,
  type ValidationSeverity,
} from "./validation_loop.ts";

// Hooks
export {
  ConversationView,
  defaultDelegationRequest,
  type AgentLifecycleHooks,
  type DelegationRequest,
  type DelegationResult,
  type IterationContext,
  type IterationDecision,
  type ToolDecision,
} from "./hooks.ts";

// Plan executor
export {
  defaultPlanExecutionConfig,
  formatPlanExecutionStatus,
  parseExecutionApprovalMode,
  PlanExecutorAgent,
  type ExecutionApprovalMode,
  type ExecutionProgress,
  type PlanExecutionConfig,
  type PlanExecutionStatus,
} from "./plan_executor.ts";

// Coordination patterns
export {
  // Contract-net
  bidScore,
  bidScoreWeighted,
  bidTimeRemaining,
  ContractNetManager,
  ContractParticipant,
  defaultTaskRequirements,
  isBiddingOpen,
  type AwardedContract,
  type BidEvaluationStrategy,
  type ContractMessage,
  type ContractTaskStatus,
  type TaskAnnouncement,
  type TaskBid,
  type TaskRequirements,

  // Saga
  CompensationReport,
  createCheckpoint,
  failureResult,
  isCompensable,
  SagaExecutor,
  successResult,
  type Checkpoint,
  type CompensableOperation,
  type CompensationOutcome,
  type CompensationStatus,
  type FileState,
  type GitCheckpoint,
  type OperationResult,
  type SagaOperationType,
  type SagaStatus,

  // Optimistic
  commitVersion,
  isCommitSuccess,
  isTokenStale,
  OptimisticController,
  type CommitResult,
  type ConflictRecord,
  type MergeStrategy,
  type OptimisticConflict,
  type OptimisticConflictDetails,
  type OptimisticStats,
  type OptimisticToken,
  type Resolution,
  type ResolutionStrategy,
  type ResourceVersion,
  // Market-based allocation
  calculateUrgency,
  createBid,
  createBudget,
  defaultPricingStrategy,
  defaultUrgencyContext,
  effectivePriority,
  isAllocated,
  MarketAllocator,
  marketBidScore,
  replenishBudget,
  type AgentBudget,
  type AllocationRecord,
  type AllocationResult,
  type CurrentHolder,
  type MarketStats,
  type MarketStatus,
  type PricingStrategy,
  type ResourceBid,
  type UrgencyContext,

  // Three-State Model
  ApplicationState,
  createOperationLog,
  defaultGitState,
  DependencyState,
  OperationState,
  ThreeStateModel,
  type ApplicationChange,
  type DependencyEdge,
  type DependencyStrength,
  type DependencyType,
  type FileStatus as ThreeStateFileStatus,
  type GitState,
  type OperationLog,
  type OperationLogStatus,
  type ProposedOperation,
  type ResourceNodeType,
  type StateChange,
  type StateSnapshot,
  type StateValidationResult,

  // Wait Queue
  fileResourceKey,
  resourceKey,
  WaitQueue,
  type QueueStatus as WaitQueueStatus,
  type RemovalReason,
  type WaiterInfo,
  type WaitQueueEvent,
  type WaitQueueHandle,
} from "./coordination/mod.ts";

// Agent Pool
export {
  AgentPool,
  type AgentPoolStats,
} from "./agent_pool.ts";

// Execution Graph
export {
  ExecutionGraph,
  telemetryFromGraph,
  type RunTelemetry,
  type StepNode,
  type ToolCallRecord,
} from "./execution_graph.ts";

// Specialized Agents
export {
  buildJudgeTaskDescription,
  extractJsonBlock,
  judgeAgentPrompt,
  parseVerdict,
  verdictHints,
  verdictType,
  formatMergeStatus,
  type JudgeAgentConfig,
  type JudgeContext,
  type JudgeVerdict,
  type MergeStatus as JudgeMergeStatus,
  type WorkerResult,
} from "./judge_agent.ts";

export {
  defaultPlannerAgentConfig,
  parsePlannerOutput,
  plannerAgentPrompt,
  validateTaskGraph,
  type DynamicTaskPriority,
  type DynamicTaskSpec,
  type PlannerAgentConfig,
  type PlannerOutput,
  type SubPlannerRequest,
} from "./planner_agent.ts";

export {
  defaultValidatorAgentConfig,
  formatValidatorStatus,
  ValidatorAgent,
  type ValidatorAgentConfig,
  type ValidatorAgentResult,
  type ValidatorAgentStatus,
} from "./validator_agent.ts";

export {
  defaultCycleOrchestratorConfig,
  type CycleOrchestratorConfig,
  type CycleOrchestratorResult,
  type CycleRecord,
  type FailurePolicy,
  type MergeStrategy as CycleMergeStrategy,
} from "./cycle_orchestrator.ts";

// MDAP - MAKER voting framework
export {
  // Error
  MdapError,
  type MdapErrorKind,
  type MdapResult,

  // Voting types
  type VotingMethod,
  type SampledResponse,
  type ResponseMetadata,
  type VoteResult,
  type EarlyStoppingConfig,
  type VotingErrorDetails,

  // Red-flag types
  type RedFlagConfig,
  type RedFlagReason,
  type RedFlagResult as MdapRedFlagResult,
  type OutputFormat,
  type RedFlagErrorDetails,

  // Subtask / Microagent types
  type Subtask as MdapSubtask,
  type SubtaskOutput as MdapSubtaskOutput,
  type MicroagentConfig,
  type MicroagentProvider,
  type MicroagentResponse,
  type SubtaskMetric,

  // Decomposition types
  type DecomposeContext,
  type CompositionFunction,
  type DecompositionResult,
  type DecompositionStrategy,
  type DecompositionErrorDetails,

  // Scaling types
  type MdapEstimate,
  type ModelCosts,
  type ConfigSummary,

  // Tool intent types
  type ToolSchema,
  type ToolIntent,
  type SubtaskOutputWithIntent,
  type ToolCategory,
  type VotingRoundMetric,

  // Composer types
  type CompositionHandler,

  // Early stopping presets
  defaultEarlyStopping,
  disabledEarlyStopping,
  aggressiveEarlyStopping,
  conservativeEarlyStopping,

  // Red-flag config presets
  strictRedFlagConfig,
  relaxedRedFlagConfig,

  // Red-flag validators
  StandardRedFlagValidator,
  AcceptAllValidator,

  // Output format helpers
  outputFormatMatches,
  outputFormatDescription,

  // Confidence extraction
  extractResponseConfidence,

  // Subtask helpers
  createAtomicSubtask,
  createSubtask,
  createSubtaskOutput,

  // Microagent config
  defaultMicroagentConfig,

  // Decomposition helpers
  defaultDecomposeContext,
  childContext,
  atomicDecomposition,
  compositeDecomposition,
  validateDecomposition,
  topologicalSort,

  // Composer
  Composer,

  // Scaling laws
  calculateKMin,
  calculatePFull,
  calculateExpectedVotes,
  estimateMdap,
  estimatePerStepSuccess,
  estimateValidResponseRate,
  calculateExpectedCost,
  suggestKForBudget,
  MODEL_COSTS,
  estimateCallCost,

  // Metrics
  MdapMetrics,

  // Tool intent helpers
  toolCategoryContains,
  readOnlyCategories,
  sideEffectCategories,
  toolSchemaToPrompt,
  parseToolIntent,

  // Borda count
  bordaCountWinner,

  // Voter
  FirstToAheadByKVoter,
  VoterBuilder,
  type RedFlagValidator,
} from "./mdap/mod.ts";

// Skills (absorbed from former @brainwires/skills package)
export { SkillExecutor, type ScriptPrepared, type SubagentPrepared } from "./skills_executor.ts";

export {
  createSkill,
  createSkillMatch,
  createSkillMetadata,
  executionMode,
  explicitMatch,
  getMetadataValue,
  hasToolRestrictions,
  inlineResult,
  isResultError,
  isScript,
  isToolAllowed,
  keywordMatch,
  parseExecutionMode,
  runsAsSubagent,
  scriptResult,
  semanticMatch,
  subagentResult,
  type MatchSource,
  type Skill,
  type SkillExecutionMode,
  type SkillMatch,
  type SkillMetadata,
  type SkillResult,
  type SkillSource,
} from "./skills_metadata.ts";

export {
  parseMetadataFromContent,
  parseSkillFile,
  parseSkillFromContent,
  parseSkillMetadata,
  renderTemplate,
  validateCompatibility,
  validateDescription,
  validateSkillName,
} from "./skills_parser.ts";

export { SkillRegistry, truncateDescription, type DiscoveryPath } from "./skills_registry.ts";

export { SkillRouter } from "./skills_router.ts";

// Agent roles (least-privilege tool restriction)
export {
  allowedTools,
  filterTools,
  roleDisplayName,
  systemPromptSuffix,
  type AgentRole,
} from "./roles.ts";

// System prompt registry (Rust-parity canonical prompts).
// Note: judgeAgentPrompt and plannerAgentPrompt already exist on this package
// from judge_agent.ts / planner_agent.ts with slightly drifted wording. Import
// the canonical versions directly from "@brainwires/agents/system_prompts/mod.ts"
// when strict Rust↔Deno parity is required.
export {
  type AgentPromptKind,
  buildAgentPrompt,
  mdapMicroagentPrompt,
  reasoningAgentPrompt,
  simpleAgentPrompt,
} from "./system_prompts/mod.ts";

// Evaluation harness (eval/)
export {
  adversarialCaseCategory,
  adversarialCategoryName,
  type AdversarialTestCase,
  type AdversarialTestType,
  AlwaysFailCase,
  AlwaysPassCase,
  ambiguousInstructionCase,
  analyzeSuiteForFaults,
  type Assertion,
  budgetExhaustionCase,
  type CategoryBaseline,
  type CategoryRegressionResult,
  computeSequenceDiff,
  type ConfidenceInterval95,
  defaultRegressionConfig,
  defaultRunOutcome,
  defaultSuiteConfig,
  type EvaluationCase,
  type EvaluationStats,
  EvaluationSuite,
  evaluationStatsFromTrials,
  evaluate as evaluateFixture,
  type ExpectedBehavior,
  failingCases,
  failingCategoryResults,
  type FaultKind,
  faultKindLabel,
  faultKindPriority,
  type FaultReport,
  faultReportPriority,
  type Fixture,
  FixtureCase,
  type FixtureMessage,
  type FixtureRunner,
  GoalPreservationCase,
  improvedCategoryResults,
  injectionPayload,
  isCiPassing,
  isExactMatch,
  levenshtein,
  loadFixtureFile,
  loadFixturesFromDir,
  LoopDetectionSimCase,
  longHorizonStabilitySuite,
  missingContextCase,
  mrr,
  ndcgAtK,
  newCapabilityFault,
  newCategoryBaseline,
  overallSuccessRate,
  percentile,
  precisionAtK,
  promptInjectionCase,
  type RegressionConfig,
  type RegressionResult,
  RegressionSuite,
  regressionFault,
  type RunOutcome,
  type SequenceDiff,
  standardAdversarialSuite,
  StochasticCase,
  type SuiteConfig,
  type SuiteResult,
  type ToolCallRecord as EvalToolCallRecord,
  ToolSequenceRecorder,
  type TrialResult,
  trialFailure,
  trialSuccess,
  trialWithMeta,
  wilsonInterval,
  withExpectRejection,
} from "./eval/mod.ts";

// SEAL (Self-Evolving Agent Loop)
export {
  asVariable as sealAsVariable,
  type BehavioralKnowledgeCache,
  type BehavioralTruth,
  compatibleTypes,
  ConfidenceStats,
  type CoreferenceRecord,
  CoreferenceResolver,
  type CorrectionRecord,
  DEFAULT_PATTERN_PROMOTION_THRESHOLD,
  DialogState,
  defaultIntegrationConfig,
  defaultReflectionConfig,
  defaultSealConfig,
  type EntityResolutionStrategy,
  type ErrorType,
  errorTypeDescription,
  FeedbackBridge,
  type FeedbackProcessingStats,
  type FilterPredicate as SealFilterPredicate,
  GlobalMemory,
  InMemoryEntityStore as SealInMemoryEntityStore,
  type IntegrationConfig,
  integrationConfigDisabled,
  integrationConfigSealToKnowledgeOnly,
  isHighConfidence,
  isLowConfidence,
  type Issue as SealIssue,
  isVariable as sealIsVariable,
  LearningCoordinator,
  type LearningStats,
  LocalMemory,
  newBehavioralTruth,
  newQueryCore,
  newSealProcessingResult,
  type PatternHint,
  type PersonalFact,
  type PersonalKnowledgeCache,
  QueryCoreExtractor,
  QueryExecutor,
  QueryPattern,
  type QueryRecord as SealQueryRecord,
  type ResolutionPattern,
  type ResolvedReference,
  type ResponseConfidence as SealResponseConfidence,
  ReflectionModule,
  ReflectionReport,
  type ReflectionConfig,
  type ScoredTruth,
  type SalienceScore,
  salienceTotal,
  SealKnowledgeCoordinator,
  SealProcessor,
  type SealConfig,
  type SealProcessingResult,
  type Severity as SealSeverity,
  severityAtLeast,
  severityCompare,
  type SuggestedFix,
  suggestedFixDescription,
  ToolErrorPattern,
  ToolStats,
  TrackedEntity,
  type TruthCategory,
  type TruthSource,
  type UnresolvedReference,
  validateIntegrationConfig,
  queryConstant,
  queryCoreToSexp,
  queryCount,
  queryJoin,
  queryResultEmpty,
  queryResultError,
  queryResultWithValues,
  queryVar,
  relationInverse,
  relationName,
  relationToEdgeType,
  type QueryCore as SealQueryCore,
  type QueryExpr as SealQueryExpr,
  type QueryOp as SealQueryOp,
  type QueryResult as SealQueryResult,
  type QueryResultValue as SealQueryResultValue,
  type QuestionType,
  type ReferenceType,
  type RelationType as SealRelationType,
  type SuperlativeDir,
} from "./seal/mod.ts";
