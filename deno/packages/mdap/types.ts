/**
 * MDAP Types - Multi-Dimensional Adaptive Planning
 *
 * Core types for the MAKER voting framework implementing:
 * - Voting: First-to-ahead-by-k consensus algorithm for error correction
 * - Microagents: Minimal context single-step agents (m=1 decomposition)
 * - Decomposition: Task decomposition strategies (binary recursive, sequential)
 * - Red Flags: Output validation and format checking
 * - Scaling: Cost/probability estimation and optimization
 * - Metrics: Execution metrics collection and reporting
 * - Composer: Result composition from subtask outputs
 * - Tool Intent: Structured tool calling intent for stateless execution
 */

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/** Voting-specific error details. */
export interface VotingErrorDetails {
  /** Which voting failure occurred (e.g. `max_samples_exceeded` when no candidate got k ahead before `maxSamples`). */
  kind:
    | "max_samples_exceeded"
    | "all_samples_red_flagged"
    | "cancelled"
    | "no_valid_responses"
    | "sampler_error"
    | "hash_error"
    | "invalid_k"
    | "parallel_error";
  /** Human-readable description of the failure. */
  message: string;
  /** Vote tally per candidate key (for max_samples_exceeded). */
  votes?: Record<string, number>;
  /** Samples drawn before giving up (for max_samples_exceeded). */
  samples?: number;
  /** Number of samples the validator rejected (for all_samples_red_flagged). */
  redFlagged?: number;
  /** Total samples drawn (for all_samples_red_flagged). */
  total?: number;
  /** Sampler calls made in the first batch that all failed (for no_valid_responses). */
  attempts?: number;
}

/** Red-flag error details. */
export interface RedFlagErrorDetails {
  /** Which red-flag check failed. */
  kind:
    | "response_too_long"
    | "invalid_format"
    | "self_correction_detected"
    | "confused_reasoning"
    | "parse_error"
    | "empty_response"
    | "invalid_json"
    | "missing_field"
    | "pattern_error";
  /** Human-readable description of the failure. */
  message: string;
}

/** Decomposition error details. */
export interface DecompositionErrorDetails {
  /** Which decomposition failure occurred (`validateDecomposition` raises `empty_result` / `invalid_dependency`, `topologicalSort` raises `circular_dependency`). */
  kind:
    | "max_depth_exceeded"
    | "cannot_decompose"
    | "circular_dependency"
    | "invalid_dependency"
    | "voting_failed"
    | "empty_result"
    | "invalid_strategy"
    | "discriminator_error";
  /** Human-readable description of the failure. */
  message: string;
}

/** All MDAP error categories. */
export type MdapErrorKind =
  | { type: "voting"; details: VotingErrorDetails }
  | { type: "red_flag"; details: RedFlagErrorDetails }
  | { type: "decomposition"; details: DecompositionErrorDetails }
  | { type: "microagent"; message: string }
  | { type: "composition"; message: string }
  | { type: "scaling"; message: string }
  | { type: "provider"; message: string }
  | { type: "config"; message: string }
  | { type: "tool_recursion_limit"; depth: number; maxDepth: number }
  | { type: "tool_execution_failed"; tool: string; reason: string }
  | { type: "tool_not_allowed"; tool: string; category: string }
  | { type: "tool_intent_parse_failed"; message: string }
  | { type: "other"; message: string };

/** Main MDAP error class. */
export class MdapError extends Error {
  /** The structured error category and its details. */
  readonly kind: MdapErrorKind;

  /**
   * Create an error whose `message` is taken from `kind.message`, else
   * `kind.details.message`, else `MDAP error: <type>`.
   */
  constructor(kind: MdapErrorKind) {
    const msg = "message" in kind
      ? (kind as { message: string }).message
      : "details" in kind
      ? (kind as { details: { message: string } }).details.message
      : `MDAP error: ${kind.type}`;
    super(msg);
    this.name = "MdapError";
    this.kind = kind;
  }

  /** Create an `{ type: "other" }` error with the given message. */
  static other(message: string): MdapError {
    return new MdapError({ type: "other", message });
  }

  /** Create a `{ type: "provider" }` error (treated as retryable) with the given message. */
  static provider(message: string): MdapError {
    return new MdapError({ type: "provider", message });
  }

  /** Check if this is a user/configuration error. */
  isUserError(): boolean {
    return this.kind.type === "config" || this.kind.type === "scaling";
  }

  /** Check if this error is retryable. */
  isRetryable(): boolean {
    return (
      this.kind.type === "provider" ||
      this.kind.type === "tool_execution_failed" ||
      (this.kind.type === "voting" &&
        this.kind.details.kind === "sampler_error") ||
      (this.kind.type === "microagent" && true)
    );
  }

  /** Check if this is a red-flag error. */
  isRedFlag(): boolean {
    return this.kind.type === "red_flag";
  }

  /** Check if this is a tool-related error. */
  isToolError(): boolean {
    return (
      this.kind.type === "tool_recursion_limit" ||
      this.kind.type === "tool_execution_failed" ||
      this.kind.type === "tool_not_allowed" ||
      this.kind.type === "tool_intent_parse_failed"
    );
  }
}

/**
 * @deprecated A leftover of the Rust `Result<T>` port: it is just `T`. Use `T`
 * directly; removed in 0.13.
 */
export type MdapResult<T> = T;

// ---------------------------------------------------------------------------
// Voting types
// ---------------------------------------------------------------------------

/** Voting method selection. */
export type VotingMethod =
  | "first_to_ahead_by_k"
  | "borda_count"
  | "confidence_weighted";

/** Response with metadata for red-flag checking. */
export interface SampledResponse<T> {
  /** The parsed/extracted value from the response. */
  value: T;
  /** Metadata about the response for validation. */
  metadata: ResponseMetadata;
  /** The raw response string (for red-flag validation). */
  rawResponse: string;
  /** Confidence score for this response (0.0 - 1.0). */
  confidence: number;
}

/** Metadata extracted from LLM response. */
export interface ResponseMetadata {
  /** Number of tokens in the response. */
  tokenCount: number;
  /** Response time in milliseconds. */
  responseTimeMs: number;
  /** Whether the format was valid (pre-red-flag check). */
  formatValid: boolean;
  /** The finish reason from the API (if available). */
  finishReason?: string;
  /** Model used for this response. */
  model?: string;
}

/** Result of the voting process. */
export interface VoteResult<T> {
  /** The winning value. */
  winner: T;
  /** Number of votes for the winner. */
  winnerVotes: number;
  /** Total number of valid votes cast. */
  totalVotes: number;
  /** Total samples taken (including red-flagged). */
  totalSamples: number;
  /** Number of red-flagged (discarded) samples. */
  redFlaggedCount: number;
  /** Distribution of votes by candidate. */
  voteDistribution: Record<string, number>;
  /** Confidence score (winnerVotes / totalVotes). */
  confidence: number;
  /** Reasons for red-flagged samples. */
  redFlagReasons: string[];
  /** Whether voting stopped early due to high confidence (RASC-style). */
  earlyStopped: boolean;
  /** Weighted confidence score (when using confidence-weighted voting). */
  weightedConfidence?: number;
  /** Voting method used. */
  votingMethod: VotingMethod;
}

/**
 * Configuration for dynamic early stopping (RASC paper: arxiv:2408.17017).
 */
export interface EarlyStoppingConfig {
  /** Minimum confidence ratio to trigger early stop (e.g., 0.85 = 85%). */
  minConfidence: number;
  /** Minimum votes before considering early stop. */
  minVotes: number;
  /** Whether early stopping is enabled. */
  enabled: boolean;
  /** Maximum variance threshold for stability-based stopping. */
  maxVarianceThreshold: number;
  /** Enable loss-of-hope detection. */
  lossOfHopeEnabled: boolean;
  /** Minimum weighted confidence for stopping. */
  minWeightedConfidence: number;
}

// ---------------------------------------------------------------------------
// Red-flag types
// ---------------------------------------------------------------------------

/** Red-flag configuration following the paper's strict approach. */
export interface RedFlagConfig {
  /** Maximum response tokens before flagging (paper: ~750). */
  maxResponseTokens: number;
  /** Require exact format match. */
  requireExactFormat: boolean;
  /** Flag responses with self-correction patterns. */
  flagSelfCorrection: boolean;
  /** Patterns indicating confused reasoning (to discard). */
  confusionPatterns: string[];
  /** Minimum response length. */
  minResponseLength: number;
  /** Maximum empty line ratio. */
  maxEmptyLineRatio: number;
}

/** Reasons for red-flagging a response. */
export type RedFlagReason =
  | { kind: "response_too_long"; tokens: number; limit: number }
  | { kind: "response_too_short"; length: number; minimum: number }
  | { kind: "invalid_format"; expected: string; got: string }
  | { kind: "self_correction_detected"; pattern: string }
  | { kind: "confused_reasoning"; pattern: string }
  | { kind: "parse_error"; message: string }
  | { kind: "empty_response" }
  | { kind: "too_many_empty_lines"; ratio: number; max: number }
  | { kind: "invalid_json"; message: string }
  | { kind: "missing_field"; field: string }
  | { kind: "truncated"; reason: string };

/** Result of red-flag validation. */
export type RedFlagResult =
  | { valid: true }
  | { valid: false; reason: RedFlagReason; severity: number };

/** Expected output format for validation. */
export type OutputFormat =
  | { kind: "exact"; value: string }
  | { kind: "pattern"; regex: string }
  | { kind: "json" }
  | { kind: "json_with_fields"; fields: string[] }
  | { kind: "markers"; start: string; end: string }
  | { kind: "one_of"; options: string[] }
  | { kind: "custom"; description: string; validatorId: string };

// ---------------------------------------------------------------------------
// Subtask / Microagent types
// ---------------------------------------------------------------------------

/** A minimal subtask that can be executed by a microagent. */
export interface Subtask {
  /** Unique identifier for this subtask. */
  id: string;
  /** Human-readable description. */
  description: string;
  /** Input state/context for this subtask. */
  inputState: unknown;
  /** Expected output format for validation. */
  expectedOutputFormat?: OutputFormat;
  /** IDs of subtasks this one depends on. */
  dependsOn: string[];
  /** Complexity estimate (0.0-1.0). */
  complexityEstimate: number;
  /** Optional specific instructions for this subtask. */
  instructions?: string;
}

/** Output from a subtask execution. */
export interface SubtaskOutput {
  /** The subtask ID this output is for. */
  subtaskId: string;
  /** The output value. */
  value: unknown;
  /** Optional next state (for stateful subtasks). */
  nextState?: unknown;
}

/** Configuration for microagent execution. */
export interface MicroagentConfig {
  /** Maximum output tokens (strict limit, paper: ~750). */
  maxOutputTokens: number;
  /** Sampling temperature. */
  temperature: number;
  /** System prompt template. */
  systemPromptTemplate: string;
  /** Red-flag configuration. */
  redFlagConfig: RedFlagConfig;
  /** Request timeout in milliseconds. */
  timeoutMs: number;
}

/** Response from a microagent provider. */
export interface MicroagentResponse {
  /** The response text. */
  text: string;
  /** Number of input tokens. */
  inputTokens: number;
  /** Number of output tokens. */
  outputTokens: number;
  /** Finish reason (if available). */
  finishReason?: string;
  /** Response time in milliseconds. */
  responseTimeMs: number;
}

/** Trait interface for providers that can be used with microagents. */
export interface MicroagentProvider {
  /**
   * Run one stateless chat completion with the given system and user prompts.
   *
   * @param system - System prompt (the filled-in `systemPromptTemplate`).
   * @param user - User message carrying the subtask input.
   * @param temperature - Sampling temperature.
   * @param maxTokens - Hard output-token limit for this call.
   * @returns The response text plus token counts and timing.
   */
  chat(
    system: string,
    user: string,
    temperature: number,
    maxTokens: number,
  ): Promise<MicroagentResponse>;

  /** Get available tools for intent expression (not execution). */
  availableTools?(): ToolSchema[];
}

// ---------------------------------------------------------------------------
// Decomposition types
// ---------------------------------------------------------------------------

/** Context for task decomposition. */
export interface DecomposeContext {
  /** Working directory for file operations. */
  workingDirectory: string;
  /** Available tools for the agent. */
  availableTools: string[];
  /** Maximum decomposition depth. */
  maxDepth: number;
  /** Current depth in recursive decomposition. */
  currentDepth: number;
  /** Additional context/constraints. */
  additionalContext?: string;
}

/** How to combine results from subtasks. */
export type CompositionFunction =
  | { kind: "identity" }
  | { kind: "concatenate" }
  | { kind: "sequence" }
  | { kind: "object_merge" }
  | { kind: "last_only" }
  | { kind: "custom"; description: string }
  | { kind: "reduce"; operation: string };

/** Result of task decomposition. */
export interface DecompositionResult {
  /** The subtasks resulting from decomposition. */
  subtasks: Subtask[];
  /** How to combine results from subtasks. */
  compositionFunction: CompositionFunction;
  /** Whether the task is already minimal. */
  isMinimal: boolean;
  /** Estimated total complexity. */
  totalComplexity: number;
}

/** Task decomposition strategy. */
export type DecompositionStrategy =
  | { kind: "binary_recursive"; maxDepth: number }
  | { kind: "simple"; maxDepth: number }
  | { kind: "sequential" }
  | { kind: "code_operations" }
  | { kind: "ai_driven"; discriminatorK: number }
  | { kind: "none" };

// ---------------------------------------------------------------------------
// Composer types
// ---------------------------------------------------------------------------

/** Trait for custom composition handlers. */
export interface CompositionHandler {
  /** Combine the ordered subtask outputs into a single composed value. */
  compose(results: SubtaskOutput[]): unknown;
}

// ---------------------------------------------------------------------------
// Scaling types
// ---------------------------------------------------------------------------

/** Cost and probability estimation result. */
export interface MdapEstimate {
  /** Expected cost in USD. */
  expectedCostUsd: number;
  /** Expected number of API calls. */
  expectedApiCalls: number;
  /** Probability of full task success. */
  successProbability: number;
  /** Recommended k value for target success rate. */
  recommendedK: number;
  /** Estimated execution time in seconds. */
  estimatedTimeSeconds: number;
  /** Per-step success probability used in calculation. */
  perStepSuccess: number;
  /** Number of steps in the task. */
  numSteps: number;
}

/** Model-specific cost estimates (per 1000 tokens). */
export interface ModelCosts {
  /** Cost per 1000 input tokens. */
  inputPer1k: number;
  /** Cost per 1000 output tokens. */
  outputPer1k: number;
}

// ---------------------------------------------------------------------------
// Metrics types
// ---------------------------------------------------------------------------

/** Summary of MDAP configuration for metrics. */
export interface ConfigSummary {
  /** The first-to-ahead-by-k margin used for voting. */
  k: number;
  /** Target probability of full-task success (0.0-1.0). */
  targetSuccessRate: number;
  /** Number of samples drawn concurrently per voting batch. */
  parallelSamples: number;
  /** Cap on samples drawn for a single subtask before voting fails. */
  maxSamplesPerSubtask: number;
  /** Name of the `DecompositionStrategy` kind in use. */
  decompositionStrategy: string;
}

/** Metrics for a single subtask execution. */
export interface SubtaskMetric {
  /** ID of the subtask these metrics describe. */
  subtaskId: string;
  /** The subtask's human-readable description. */
  description: string;
  /** Total samples drawn for this subtask, including red-flagged ones. */
  samplesNeeded: number;
  /** Number of samples the red-flag validator rejected. */
  redFlagsHit: number;
  /** Formatted reason string for each red-flagged sample. */
  redFlagReasons: string[];
  /** Winner's share of valid votes (`winnerVotes / totalVotes`). */
  finalConfidence: number;
  /** Wall-clock time spent on this subtask in milliseconds. */
  executionTimeMs: number;
  /** Votes received by the winning candidate. */
  winnerVotes: number;
  /** Valid (non-red-flagged) votes cast. */
  totalVotes: number;
  /** Whether voting produced a winner. */
  succeeded: boolean;
  /** Input tokens consumed across all samples. */
  inputTokens: number;
  /** Output tokens produced across all samples. */
  outputTokens: number;
  /** The subtask's `complexityEstimate` (0.0-1.0). */
  complexityEstimate: number;
}

/** Metrics for a single voting round. */
export interface VotingRoundMetric {
  /** Zero-based index of the step (subtask) being voted on. */
  step: number;
  /** Zero-based index of the sampling batch within the step. */
  round: number;
  /** Vote tally per candidate key after this round. */
  candidates: Record<string, number>;
  /** Candidate key that won in this round, if voting concluded. */
  winner?: string;
  /** Samples rejected by the red-flag validator in this round. */
  redFlaggedThisRound: number;
  /** Time spent sampling and validating this round in milliseconds. */
  roundTimeMs: number;
}

// ---------------------------------------------------------------------------
// Tool intent types
// ---------------------------------------------------------------------------

/** Schema describing a tool's interface for intent expression. */
export interface ToolSchema {
  /** Tool name. */
  name: string;
  /** Description of what the tool does. */
  description: string;
  /** Parameter descriptions (name -> description). */
  parameters: Record<string, string>;
  /** Required parameters. */
  required: string[];
  /** Tool category. */
  category?: ToolCategory;
}

/** A tool call intent that can be voted on. */
export interface ToolIntent {
  /** Tool name to call. */
  toolName: string;
  /** Tool arguments as JSON. */
  arguments: unknown;
  /** Why this tool is needed (for debugging/logging). */
  rationale?: string;
}

/** Extended subtask output that may include tool intent. */
export interface SubtaskOutputWithIntent {
  /** Base subtask output. */
  output: SubtaskOutput;
  /** Optional tool intent. */
  toolIntent?: ToolIntent;
  /** Whether the output is complete or waiting for tool result. */
  awaitingToolResult: boolean;
}

/** Tool categories for permission control. */
export type ToolCategory =
  | "file_read"
  | "file_write"
  | "search"
  | "semantic_search"
  | "bash"
  | "git"
  | "web"
  | "agent_pool"
  | "task_manager"
  | "mcp"
  | { custom: string };
