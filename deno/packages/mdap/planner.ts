/**
 * MDAP Planner - Plan generation, scoring, decomposition, composition, scaling,
 * red-flag validation, and metrics collection.
 *
 * Implements the MAKER paper's approach to reliable agent execution through
 * task decomposition, microagent execution, result composition, and scaling laws.
 */

import type {
  CompositionFunction,
  CompositionHandler,
  ConfigSummary,
  DecomposeContext,
  DecompositionResult,
  EarlyStoppingConfig,
  MdapEstimate,
  MicroagentConfig,
  ModelCosts,
  OutputFormat,
  RedFlagConfig,
  RedFlagResult,
  ResponseMetadata,
  Subtask,
  SubtaskMetric,
  SubtaskOutput,
  ToolCategory,
  ToolIntent,
  ToolSchema,
  VotingRoundMetric,
} from "./types.ts";
import { MdapError } from "./types.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_MIN_CONFIDENCE = 0.85;
const DEFAULT_MIN_VOTES = 3;
const DEFAULT_MAX_VARIANCE_THRESHOLD = 0.15;
const DEFAULT_MIN_WEIGHTED_CONFIDENCE = 0.8;

const AGGRESSIVE_MIN_CONFIDENCE = 0.75;
const AGGRESSIVE_MIN_VOTES = 2;
const AGGRESSIVE_MAX_VARIANCE_THRESHOLD = 0.2;
const AGGRESSIVE_MIN_WEIGHTED_CONFIDENCE = 0.7;

const CONSERVATIVE_MIN_CONFIDENCE = 0.9;
const CONSERVATIVE_MIN_VOTES = 5;
const CONSERVATIVE_MAX_VARIANCE_THRESHOLD = 0.1;
const CONSERVATIVE_MIN_WEIGHTED_CONFIDENCE = 0.85;

const _DEFAULT_BATCH_SIZE = 4;

// ---------------------------------------------------------------------------
// Early stopping presets
// ---------------------------------------------------------------------------

/** Create default early stopping config. */
export function defaultEarlyStopping(): EarlyStoppingConfig {
  return {
    minConfidence: DEFAULT_MIN_CONFIDENCE,
    minVotes: DEFAULT_MIN_VOTES,
    enabled: true,
    maxVarianceThreshold: DEFAULT_MAX_VARIANCE_THRESHOLD,
    lossOfHopeEnabled: true,
    minWeightedConfidence: DEFAULT_MIN_WEIGHTED_CONFIDENCE,
  };
}

/** Create disabled early stopping config. */
export function disabledEarlyStopping(): EarlyStoppingConfig {
  return {
    ...defaultEarlyStopping(),
    enabled: false,
    lossOfHopeEnabled: false,
  };
}

/** Create aggressive early stopping config. */
export function aggressiveEarlyStopping(): EarlyStoppingConfig {
  return {
    minConfidence: AGGRESSIVE_MIN_CONFIDENCE,
    minVotes: AGGRESSIVE_MIN_VOTES,
    enabled: true,
    maxVarianceThreshold: AGGRESSIVE_MAX_VARIANCE_THRESHOLD,
    lossOfHopeEnabled: true,
    minWeightedConfidence: AGGRESSIVE_MIN_WEIGHTED_CONFIDENCE,
  };
}

/** Create conservative early stopping config. */
export function conservativeEarlyStopping(): EarlyStoppingConfig {
  return {
    minConfidence: CONSERVATIVE_MIN_CONFIDENCE,
    minVotes: CONSERVATIVE_MIN_VOTES,
    enabled: true,
    maxVarianceThreshold: CONSERVATIVE_MAX_VARIANCE_THRESHOLD,
    lossOfHopeEnabled: true,
    minWeightedConfidence: CONSERVATIVE_MIN_WEIGHTED_CONFIDENCE,
  };
}

// ---------------------------------------------------------------------------
// Red-flag configuration presets
// ---------------------------------------------------------------------------

const STRICT_CONFUSION_PATTERNS = [
  "Wait,",
  "Actually,",
  "Let me reconsider",
  "I made a mistake",
  "On second thought",
  "Hmm,",
  "I think I",
  "Let me correct",
  "Sorry, I meant",
  "That's not right",
];

/** Paper's strict red-flag configuration. */
export function strictRedFlagConfig(): RedFlagConfig {
  return {
    maxResponseTokens: 750,
    requireExactFormat: true,
    flagSelfCorrection: true,
    confusionPatterns: [...STRICT_CONFUSION_PATTERNS],
    minResponseLength: 1,
    maxEmptyLineRatio: 0.5,
  };
}

/** Relaxed red-flag configuration. */
export function relaxedRedFlagConfig(): RedFlagConfig {
  return {
    maxResponseTokens: 1500,
    requireExactFormat: false,
    flagSelfCorrection: false,
    confusionPatterns: [],
    minResponseLength: 0,
    maxEmptyLineRatio: 0.8,
  };
}

// ---------------------------------------------------------------------------
// Output format helpers
// ---------------------------------------------------------------------------

/** Check if a response matches an expected output format. */
export function outputFormatMatches(
  format: OutputFormat,
  response: string,
): boolean {
  const trimmed = response.trim();
  switch (format.kind) {
    case "exact":
      return trimmed === format.value.trim();
    case "pattern":
      return new RegExp(format.regex).test(trimmed);
    case "json":
      try {
        JSON.parse(trimmed);
        return true;
      } catch {
        return false;
      }
    case "json_with_fields": {
      try {
        const obj = JSON.parse(trimmed);
        if (typeof obj !== "object" || obj === null) return false;
        return format.fields.every((f) => f in obj);
      } catch {
        return false;
      }
    }
    case "markers":
      return trimmed.includes(format.start) && trimmed.includes(format.end);
    case "one_of":
      return format.options.some((o) => trimmed === o.trim());
    case "custom":
      return true; // external validator needed
  }
}

/** Get a description of an output format. */
export function outputFormatDescription(format: OutputFormat): string {
  switch (format.kind) {
    case "exact":
      return `exact: '${format.value}'`;
    case "pattern":
      return `pattern: ${format.regex}`;
    case "json":
      return "valid JSON";
    case "json_with_fields":
      return `JSON with fields: ${format.fields.join(", ")}`;
    case "markers":
      return `markers: ${format.start}...${format.end}`;
    case "one_of":
      return `one of: ${format.options.join(", ")}`;
    case "custom":
      return format.description;
  }
}

// ---------------------------------------------------------------------------
// Red-flag validator
// ---------------------------------------------------------------------------

/** Standard red-flag validator implementing the paper's approach. */
export class StandardRedFlagValidator {
  /** Thresholds and pattern lists the checks are evaluated against. */
  readonly config: RedFlagConfig;
  /** Format the response must match when `config.requireExactFormat` is set. */
  readonly expectedFormat?: OutputFormat;

  /**
   * Create a validator with explicit thresholds.
   *
   * @param config - Red-flag thresholds (see `strictRedFlagConfig` / `relaxedRedFlagConfig`).
   * @param expectedFormat - Optional output format enforced by the format check.
   */
  constructor(config: RedFlagConfig, expectedFormat?: OutputFormat) {
    this.config = config;
    this.expectedFormat = expectedFormat;
  }

  /** Validator using `strictRedFlagConfig()` and no expected format. */
  static strict(): StandardRedFlagValidator {
    return new StandardRedFlagValidator(strictRedFlagConfig());
  }

  /** Validator using `strictRedFlagConfig()` that also enforces `format`. */
  static withFormat(format: OutputFormat): StandardRedFlagValidator {
    return new StandardRedFlagValidator(strictRedFlagConfig(), format);
  }

  /**
   * Run the checks in order — length, truncation, format, self-correction,
   * empty-line ratio — and return the first failure, or `{ valid: true }`.
   *
   * @param response - Raw model output.
   * @param metadata - Token count and finish reason used by the length and truncation checks.
   */
  validate(response: string, metadata: ResponseMetadata): RedFlagResult {
    // 1. Check length constraints
    const lengthResult = this.checkLength(response, metadata);
    if (lengthResult) return lengthResult;

    // 2. Check truncation
    const truncResult = this.checkTruncation(metadata);
    if (truncResult) return truncResult;

    // 3. Check format
    const formatResult = this.checkFormat(response);
    if (formatResult) return formatResult;

    // 4. Check self-correction patterns
    const correctionResult = this.checkSelfCorrection(response);
    if (correctionResult) return correctionResult;

    // 5. Check empty line ratio
    const emptyResult = this.checkEmptyLines(response);
    if (emptyResult) return emptyResult;

    return { valid: true };
  }

  /** Flag empty (severity 1.0), too-short (0.9) or over-token-limit (0.8) responses. */
  private checkLength(
    response: string,
    metadata: ResponseMetadata,
  ): RedFlagResult | null {
    if (response.trim().length === 0) {
      return {
        valid: false,
        reason: { kind: "empty_response" },
        severity: 1.0,
      };
    }
    if (response.length < this.config.minResponseLength) {
      return {
        valid: false,
        reason: {
          kind: "response_too_short",
          length: response.length,
          minimum: this.config.minResponseLength,
        },
        severity: 0.9,
      };
    }
    if (metadata.tokenCount > this.config.maxResponseTokens) {
      return {
        valid: false,
        reason: {
          kind: "response_too_long",
          tokens: metadata.tokenCount,
          limit: this.config.maxResponseTokens,
        },
        severity: 0.8,
      };
    }
    return null;
  }

  /** Flag a finish reason containing "length" or "max_tokens" as truncated (severity 0.85). */
  private checkTruncation(metadata: ResponseMetadata): RedFlagResult | null {
    if (metadata.finishReason) {
      const lower = metadata.finishReason.toLowerCase();
      if (lower.includes("length") || lower.includes("max_tokens")) {
        return {
          valid: false,
          reason: { kind: "truncated", reason: metadata.finishReason },
          severity: 0.85,
        };
      }
    }
    return null;
  }

  /** Flag a mismatch against `expectedFormat` (severity 0.9) when exact format is required. */
  private checkFormat(response: string): RedFlagResult | null {
    if (!this.config.requireExactFormat || !this.expectedFormat) return null;
    if (!outputFormatMatches(this.expectedFormat, response)) {
      return {
        valid: false,
        reason: {
          kind: "invalid_format",
          expected: outputFormatDescription(this.expectedFormat),
          got: response.trim().slice(0, 50),
        },
        severity: 0.9,
      };
    }
    return null;
  }

  /** Flag the first `confusionPatterns` substring found (severity 0.7) when enabled. */
  private checkSelfCorrection(response: string): RedFlagResult | null {
    if (!this.config.flagSelfCorrection) return null;
    for (const pattern of this.config.confusionPatterns) {
      if (response.includes(pattern)) {
        return {
          valid: false,
          reason: { kind: "self_correction_detected", pattern },
          severity: 0.7,
        };
      }
    }
    return null;
  }

  /** Flag a blank-line ratio above `maxEmptyLineRatio` (severity 0.6). */
  private checkEmptyLines(response: string): RedFlagResult | null {
    const lines = response.split("\n");
    if (lines.length === 0) return null;
    const emptyCount = lines.filter((l) => l.trim().length === 0).length;
    const ratio = emptyCount / lines.length;
    if (ratio > this.config.maxEmptyLineRatio) {
      return {
        valid: false,
        reason: {
          kind: "too_many_empty_lines",
          ratio,
          max: this.config.maxEmptyLineRatio,
        },
        severity: 0.6,
      };
    }
    return null;
  }
}

/** Always-accept validator for testing. */
export class AcceptAllValidator {
  /** Always returns `{ valid: true }` regardless of input. */
  validate(_response: string, _metadata: ResponseMetadata): RedFlagResult {
    return { valid: true };
  }
}

// ---------------------------------------------------------------------------
// Confidence extraction (CISC paper: arxiv:2502.06233v1)
// ---------------------------------------------------------------------------

/** Extract confidence from a microagent response. */
export function extractResponseConfidence(
  text: string,
  metadata: ResponseMetadata,
): number {
  let confidence = 0.75;

  // 1. Finish reason
  if (
    metadata.finishReason === "stop" ||
    metadata.finishReason === "end_turn"
  ) {
    confidence += 0.1;
  } else if (
    metadata.finishReason === "length" ||
    metadata.finishReason === "max_tokens"
  ) {
    confidence -= 0.25;
  }

  // 2. Response length
  if (metadata.tokenCount < 10) {
    confidence -= 0.2;
  } else if (metadata.tokenCount > 700) {
    confidence -= 0.15;
  }

  // 3. Hedging patterns
  const lower = text.toLowerCase();
  const hedging = [
    "i'm not sure",
    "i think",
    "possibly",
    "might be",
    "could be",
    "probably",
    "perhaps",
    "maybe",
    "unclear",
    "i guess",
  ];
  const hedgingCount = hedging.filter((p) => lower.includes(p)).length;
  confidence -= Math.min(hedgingCount * 0.08, 0.3);

  // 4. Self-correction patterns
  const corrections = [
    "wait,",
    "actually,",
    "let me reconsider",
    "i made a mistake",
    "correction:",
    "i was wrong",
    "on second thought",
  ];
  const correctionCount = corrections.filter((p) => lower.includes(p)).length;
  confidence -= Math.min(correctionCount * 0.15, 0.3);

  // 5. Confident assertion patterns
  const confident = [
    "the answer is",
    "definitely",
    "certainly",
    "clearly",
    "the solution is",
    "this will work",
  ];
  const confidentCount = confident.filter((p) => lower.includes(p)).length;
  confidence += Math.min(confidentCount * 0.05, 0.1);

  // 6. Format validity
  if (!metadata.formatValid) {
    confidence -= 0.2;
  }

  return Math.max(0.1, Math.min(0.99, confidence));
}

// ---------------------------------------------------------------------------
// Subtask helpers
// ---------------------------------------------------------------------------

/** Create an atomic subtask. */
export function createAtomicSubtask(description: string): Subtask {
  return {
    id: crypto.randomUUID(),
    description,
    inputState: null,
    dependsOn: [],
    complexityEstimate: 0.5,
  };
}

/** Create a subtask with full configuration. */
export function createSubtask(
  id: string,
  description: string,
  inputState: unknown = null,
): Subtask {
  return {
    id,
    description,
    inputState,
    dependsOn: [],
    complexityEstimate: 0.5,
  };
}

/** Create a subtask output. */
export function createSubtaskOutput(
  subtaskId: string,
  value: unknown,
): SubtaskOutput {
  return { subtaskId, value };
}

// ---------------------------------------------------------------------------
// Microagent config
// ---------------------------------------------------------------------------

const MICROAGENT_SYSTEM_PROMPT =
  `You are a focused execution agent. Your job is to complete ONE specific subtask.

RULES:
1. Complete ONLY the specified subtask - nothing more, nothing less
2. Output ONLY the requested format - no explanations unless required
3. If you're unsure, output your best answer - do NOT hedge or explain uncertainty
4. Do NOT use phrases like "Wait,", "Actually,", "Let me reconsider" - just give the answer
5. Be concise and direct

Your subtask: {subtask_description}
Expected output format: {output_format}`;

/** Create default microagent configuration. */
export function defaultMicroagentConfig(): MicroagentConfig {
  return {
    maxOutputTokens: 750,
    temperature: 0.1,
    systemPromptTemplate: MICROAGENT_SYSTEM_PROMPT,
    redFlagConfig: strictRedFlagConfig(),
    timeoutMs: 30000,
  };
}

// ---------------------------------------------------------------------------
// Decomposition helpers
// ---------------------------------------------------------------------------

/** Create a default decomposition context. */
export function defaultDecomposeContext(
  workingDirectory = ".",
): DecomposeContext {
  return {
    workingDirectory,
    availableTools: [],
    maxDepth: 10,
    currentDepth: 0,
  };
}

/** Create a child decomposition context (increment depth). */
export function childContext(ctx: DecomposeContext): DecomposeContext {
  return { ...ctx, currentDepth: ctx.currentDepth + 1 };
}

/** Create an atomic decomposition result. */
export function atomicDecomposition(subtask: Subtask): DecompositionResult {
  return {
    subtasks: [subtask],
    compositionFunction: { kind: "identity" },
    isMinimal: true,
    totalComplexity: subtask.complexityEstimate,
  };
}

/** Create a composite decomposition result. */
export function compositeDecomposition(
  subtasks: Subtask[],
  compositionFunction: CompositionFunction,
): DecompositionResult {
  return {
    subtasks,
    compositionFunction,
    isMinimal: false,
    totalComplexity: subtasks.reduce(
      (sum, s) => sum + s.complexityEstimate,
      0,
    ),
  };
}

/** Validate a decomposition result. */
export function validateDecomposition(result: DecompositionResult): void {
  if (result.subtasks.length === 0) {
    throw new MdapError({
      type: "decomposition",
      details: {
        kind: "empty_result",
        message: "Decomposition produced no subtasks",
      },
    });
  }
  const ids = new Set(result.subtasks.map((s) => s.id));
  for (const subtask of result.subtasks) {
    for (const dep of subtask.dependsOn) {
      if (!ids.has(dep)) {
        throw new MdapError({
          type: "decomposition",
          details: {
            kind: "invalid_dependency",
            message: `Subtask '${subtask.id}' depends on non-existent '${dep}'`,
          },
        });
      }
    }
  }
}

/** Topological sort of subtasks by dependencies. */
export function topologicalSort(subtasks: Subtask[]): Subtask[] {
  const inDegree = new Map<string, number>();
  const graph = new Map<string, string[]>();
  const subtaskMap = new Map<string, Subtask>();

  for (const s of subtasks) {
    inDegree.set(s.id, s.dependsOn.length);
    graph.set(s.id, []);
    subtaskMap.set(s.id, s);
  }

  for (const s of subtasks) {
    for (const dep of s.dependsOn) {
      graph.get(dep)?.push(s.id);
    }
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const result: Subtask[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    const s = subtaskMap.get(id);
    if (s) result.push(s);
    for (const dependent of graph.get(id) ?? []) {
      const deg = (inDegree.get(dependent) ?? 1) - 1;
      inDegree.set(dependent, deg);
      if (deg === 0) queue.push(dependent);
    }
  }

  if (result.length !== subtasks.length) {
    throw new MdapError({
      type: "decomposition",
      details: {
        kind: "circular_dependency",
        message: "Circular dependency detected in subtasks",
      },
    });
  }

  return result;
}

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------

/** Result composer for combining subtask outputs. */
export class Composer {
  private customHandlers = new Map<string, CompositionHandler>();

  /**
   * Register a handler invoked for `{ kind: "custom" }` functions whose
   * `description` equals `name`.
   */
  registerHandler(name: string, handler: CompositionHandler): void {
    this.customHandlers.set(name, handler);
  }

  /**
   * Combine subtask outputs according to `fn`.
   *
   * @returns The composed value; for `custom` without a registered handler,
   * `{ composition, results }`.
   * @throws {MdapError} `composition` when `results` is empty, the reduce
   * operation is unknown, or a value cannot be coerced.
   */
  compose(results: SubtaskOutput[], fn: CompositionFunction): unknown {
    if (results.length === 0) {
      throw new MdapError({
        type: "composition",
        message: "No results to compose",
      });
    }

    switch (fn.kind) {
      case "identity":
        return results[0].value;
      case "concatenate":
        return this.concatenate(results);
      case "sequence":
        return results.map((r) => r.value);
      case "object_merge":
        return this.objectMerge(results);
      case "last_only":
        return results[results.length - 1].value;
      case "custom":
        return this.customCompose(results, fn.description);
      case "reduce":
        return this.reduce(results, fn.operation);
    }
  }

  /** Join values with newlines; array values are flattened one element per line. */
  private concatenate(results: SubtaskOutput[]): string {
    return results
      .map((r) => {
        const v = r.value;
        if (typeof v === "string") return v;
        if (Array.isArray(v)) return v.map(String).join("\n");
        return String(v);
      })
      .join("\n");
  }

  /** Shallow-merge plain-object values; non-objects are keyed by their `subtaskId`. */
  private objectMerge(results: SubtaskOutput[]): Record<string, unknown> {
    const map: Record<string, unknown> = {};
    for (const r of results) {
      if (
        typeof r.value === "object" &&
        r.value !== null &&
        !Array.isArray(r.value)
      ) {
        Object.assign(map, r.value);
      } else {
        map[r.subtaskId] = r.value;
      }
    }
    return map;
  }

  /**
   * Fold values with a named operation: `sum`/`add`, `multiply`/`product`,
   * `max`, `min`, `and`/`all`, `or`/`any`, `concat`/`join` (case-insensitive).
   */
  private reduce(results: SubtaskOutput[], operation: string): unknown {
    const op = operation.toLowerCase();
    switch (op) {
      case "sum":
      case "add":
        return results.reduce(
          (acc, r) => acc + this.extractNumber(r.value),
          0,
        );
      case "multiply":
      case "product":
        return results.reduce(
          (acc, r) => acc * this.extractNumber(r.value),
          1,
        );
      case "max":
        return Math.max(...results.map((r) => this.extractNumber(r.value)));
      case "min":
        return Math.min(...results.map((r) => this.extractNumber(r.value)));
      case "and":
      case "all":
        return results.every((r) => this.extractBool(r.value));
      case "or":
      case "any":
        return results.some((r) => this.extractBool(r.value));
      case "concat":
      case "join":
        return this.concatenate(results);
      default:
        throw new MdapError({
          type: "composition",
          message: `Unknown reduce operation: ${operation}`,
        });
    }
  }

  /** Dispatch to a registered handler, or wrap the raw values when none matches. */
  private customCompose(
    results: SubtaskOutput[],
    description: string,
  ): unknown {
    const handler = this.customHandlers.get(description);
    if (handler) return handler.compose(results);
    return {
      composition: description,
      results: results.map((r) => r.value),
    };
  }

  /** Coerce a number or numeric string; throws `composition` otherwise. */
  private extractNumber(value: unknown): number {
    if (typeof value === "number") return value;
    if (typeof value === "string") {
      const n = parseFloat(value);
      if (!isNaN(n)) return n;
    }
    throw new MdapError({
      type: "composition",
      message: `Cannot extract number from ${typeof value}`,
    });
  }

  /** Coerce a boolean, "true/yes/1" / "false/no/0" string, or non-zero number; throws `composition` otherwise. */
  private extractBool(value: unknown): boolean {
    if (typeof value === "boolean") return value;
    if (typeof value === "string") {
      const lower = value.toLowerCase();
      if (["true", "yes", "1"].includes(lower)) return true;
      if (["false", "no", "0"].includes(lower)) return false;
    }
    if (typeof value === "number") return value !== 0;
    throw new MdapError({
      type: "composition",
      message: `Cannot extract bool from ${typeof value}`,
    });
  }
}

// ---------------------------------------------------------------------------
// Scaling laws (MAKER paper equations 13-18)
// ---------------------------------------------------------------------------

/**
 * Calculate minimum k for target success probability.
 * Equation 14: k_min = ceil(ln(t^(-1/s) - 1) / ln((1-p)/p))
 */
export function calculateKMin(
  numSteps: number,
  p: number,
  target: number,
): number {
  if (p <= 0.5) return Number.MAX_SAFE_INTEGER;

  const ratio = (1 - p) / p;

  if (target >= 0.9999) {
    const a = Math.pow(target, -1 / numSteps) - 1;
    if (a <= 0 || ratio <= 0) return 10;
    const k = Math.ceil(Math.log(a) / Math.log(ratio));
    return Math.max(1, Math.min(100, k));
  }

  const a = Math.pow(target, -1 / numSteps) - 1;
  if (a <= 0) return 1;
  if (ratio <= 0 || ratio >= 1) return 1;

  return Math.max(1, Math.ceil(Math.log(a) / Math.log(ratio)));
}

/**
 * Calculate full-task success probability.
 * Equation 13: p_full = (1 + ((1-p)/p)^k)^(-s)
 */
export function calculatePFull(numSteps: number, p: number, k: number): number {
  if (p <= 0.5) return 0;
  const ratio = (1 - p) / p;
  const ratioK = k > 50 ? 0 : Math.pow(ratio, k);
  const pSub = 1 / (1 + ratioK);
  return Math.pow(pSub, numSteps);
}

/** Calculate expected number of votes per step. */
export function calculateExpectedVotes(p: number, k: number): number {
  if (p <= 0.5) return Infinity;
  return k / (2 * p - 1);
}

/**
 * Estimate MDAP execution cost and success probability.
 * Implements equations 13-18 from the MAKER paper.
 */
export function estimateMdap(
  numSteps: number,
  perStepSuccessRate: number,
  validResponseRate: number,
  costPerSampleUsd: number,
  targetSuccessRate: number,
): MdapEstimate {
  if (numSteps === 0) {
    throw new MdapError({
      type: "scaling",
      message: "Invalid step count: must be > 0",
    });
  }
  if (perStepSuccessRate <= 0.5) {
    throw new MdapError({
      type: "scaling",
      message:
        `Voting cannot converge: per-step success rate ${perStepSuccessRate} <= 0.5`,
    });
  }
  if (perStepSuccessRate >= 1.0) {
    throw new MdapError({
      type: "scaling",
      message: `Invalid success probability: ${perStepSuccessRate}`,
    });
  }
  if (targetSuccessRate <= 0 || targetSuccessRate >= 1.0) {
    throw new MdapError({
      type: "scaling",
      message: `Invalid target probability: ${targetSuccessRate}`,
    });
  }

  const s = numSteps;
  const p = perStepSuccessRate;
  const t = targetSuccessRate;
  const v = Math.max(0.01, Math.min(1.0, validResponseRate));
  const c = costPerSampleUsd;

  const recommendedK = calculateKMin(s, p, t);
  const successProbability = calculatePFull(s, p, recommendedK);
  const expectedCostUsd = (c * s * recommendedK) / (v * (2 * p - 1));
  const expectedApiCalls = Math.ceil((s * recommendedK) / v);
  const timePerStep = 0.5 * Math.ceil(recommendedK / 4);
  const estimatedTimeSeconds = s * timePerStep;

  return {
    expectedCostUsd,
    expectedApiCalls,
    successProbability,
    recommendedK,
    estimatedTimeSeconds,
    perStepSuccess: p,
    numSteps,
  };
}

/** Estimate per-step success rate from sample data. */
export function estimatePerStepSuccess(
  totalSamples: number,
  correctSamples: number,
  redFlaggedSamples: number,
): number {
  const valid = Math.max(0, totalSamples - redFlaggedSamples);
  if (valid === 0) return 0.5;
  return Math.max(0, Math.min(1, correctSamples / valid));
}

/** Estimate valid response rate from sample data. */
export function estimateValidResponseRate(
  totalSamples: number,
  redFlaggedSamples: number,
): number {
  if (totalSamples === 0) return 0.95;
  const valid = Math.max(0, totalSamples - redFlaggedSamples);
  return Math.max(0.01, Math.min(1, valid / totalSamples));
}

/** Calculate expected cost for a specific configuration. */
export function calculateExpectedCost(
  numSteps: number,
  k: number,
  validRate: number,
  perStepSuccess: number,
  costPerCall: number,
): number {
  const v = Math.max(0.01, Math.min(1.0, validRate));
  const p = Math.max(0.51, Math.min(0.999, perStepSuccess));
  return (costPerCall * numSteps * k) / (v * (2 * p - 1));
}

/** Suggest optimal k for budget constraint. */
export function suggestKForBudget(
  numSteps: number,
  perStepSuccess: number,
  validRate: number,
  costPerCall: number,
  budgetUsd: number,
): number {
  const v = Math.max(0.01, Math.min(1.0, validRate));
  const p = Math.max(0.51, Math.min(0.999, perStepSuccess));
  const k = (budgetUsd * v * (2 * p - 1)) / (costPerCall * numSteps);
  return Math.max(1, Math.floor(k));
}

/** Model cost presets. */
export const MODEL_COSTS: Record<string, ModelCosts> = {
  claudeSonnet: { inputPer1k: 0.003, outputPer1k: 0.015 },
  claudeHaiku: { inputPer1k: 0.00025, outputPer1k: 0.00125 },
  gpt4o: { inputPer1k: 0.0025, outputPer1k: 0.01 },
  gpt4oMini: { inputPer1k: 0.00015, outputPer1k: 0.0006 },
};

/** Estimate cost for a single call. */
export function estimateCallCost(
  costs: ModelCosts,
  inputTokens: number,
  outputTokens: number,
): number {
  return (
    (inputTokens / 1000) * costs.inputPer1k +
    (outputTokens / 1000) * costs.outputPer1k
  );
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/** Comprehensive MDAP execution metrics. */
export class MdapMetrics {
  /** Identifier of the execution these metrics belong to. */
  executionId: string;
  /** Epoch-millisecond start timestamp; set by the constructor and `start()`. */
  startTime?: number;
  /** Epoch-millisecond end timestamp; set by `finalize()`. */
  endTime?: number;
  /** Snapshot of the configuration used, when created via `withConfig`. */
  configSummary?: ConfigSummary;

  /** Per-subtask metrics in the order they were recorded. */
  subtaskMetrics: SubtaskMetric[] = [];
  /** Planned step count; set by the caller, used to compute `actualSuccessRate`. */
  totalSteps = 0;
  /** Subtasks recorded with `succeeded: true`. */
  completedSteps = 0;
  /** Subtasks recorded with `succeeded: false`. */
  failedSteps = 0;

  /** Sum of `samplesNeeded` across recorded subtasks. */
  totalSamples = 0;
  /** Sum of `totalVotes` (non-red-flagged samples) across recorded subtasks. */
  validSamples = 0;
  /** Sum of `redFlagsHit` across recorded subtasks. */
  redFlaggedSamples = 0;
  /** Occurrence count per red-flag reason string. */
  redFlagBreakdown: Record<string, number> = {};

  /** Voting rounds recorded via `recordVotingRound`. */
  votingRounds: VotingRoundMetric[] = [];
  /** Mean `totalVotes` per completed step; computed by `finalize()`. */
  averageVotesPerStep = 0;
  /** Largest `totalVotes` seen for a single subtask. */
  maxVotesForSingleStep = 0;
  /** Smallest non-zero `totalVotes` seen for a single subtask (0 after `finalize()` if none). */
  minVotesForSingleStep: number = Infinity;

  /** Accumulated spend from `addSampleCost`, in USD. */
  actualCostUsd = 0;
  /** Caller-supplied up-front cost estimate, in USD. */
  estimatedCostUsd = 0;
  /** `actualCostUsd / completedSteps`; computed by `finalize()`. */
  costPerStep = 0;
  /** Sum of input tokens across recorded subtasks. */
  totalInputTokens = 0;
  /** Sum of output tokens across recorded subtasks. */
  totalOutputTokens = 0;

  /** Wall-clock seconds between `startTime` and `endTime`; computed by `finalize()`. */
  totalTimeSeconds = 0;
  /** Mean `executionTimeMs` per completed step; computed by `finalize()`. */
  averageTimePerStepMs = 0;
  /** Caller-supplied seconds spent in voting. */
  votingTimeSeconds = 0;
  /** Caller-supplied seconds spent in decomposition. */
  decompositionTimeSeconds = 0;

  /** Overall outcome passed to `finalize()`. */
  finalSuccess = false;
  /** Caller-supplied a-priori success probability (see `estimateMdap`). */
  estimatedSuccessProbability = 0;
  /** `completedSteps / totalSteps`; computed by `finalize()` when `totalSteps > 0`. */
  actualSuccessRate = 0;

  /** Model identifier, if the caller records it. */
  model?: string;
  /** Provider identifier, if the caller records it. */
  provider?: string;

  /** Create metrics for `executionId` with `startTime` set to now. */
  constructor(executionId: string) {
    this.executionId = executionId;
    this.startTime = Date.now();
  }

  /** Create metrics with `configSummary` pre-populated. */
  static withConfig(executionId: string, config: ConfigSummary): MdapMetrics {
    const m = new MdapMetrics(executionId);
    m.configSummary = config;
    return m;
  }

  /** Reset `startTime` to now. */
  start(): void {
    this.startTime = Date.now();
  }

  /**
   * Fold one subtask's metrics into the running totals (samples, tokens,
   * completed/failed counts, min/max votes, red-flag breakdown) and append it
   * to `subtaskMetrics`.
   */
  recordSubtask(metric: SubtaskMetric): void {
    this.totalSamples += metric.samplesNeeded;
    this.redFlaggedSamples += metric.redFlagsHit;
    this.validSamples += metric.totalVotes;
    this.totalInputTokens += metric.inputTokens;
    this.totalOutputTokens += metric.outputTokens;

    if (metric.succeeded) {
      this.completedSteps++;
    } else {
      this.failedSteps++;
    }

    if (metric.totalVotes > 0) {
      this.maxVotesForSingleStep = Math.max(
        this.maxVotesForSingleStep,
        metric.totalVotes,
      );
      this.minVotesForSingleStep = Math.min(
        this.minVotesForSingleStep,
        metric.totalVotes,
      );
    }

    for (const reason of metric.redFlagReasons) {
      this.redFlagBreakdown[reason] = (this.redFlagBreakdown[reason] ?? 0) + 1;
    }

    this.subtaskMetrics.push(metric);
  }

  /** Append a voting-round record to `votingRounds`. */
  recordVotingRound(round: VotingRoundMetric): void {
    this.votingRounds.push(round);
  }

  /** Add the cost of one sample to `actualCostUsd`. */
  addSampleCost(costUsd: number): void {
    this.actualCostUsd += costUsd;
  }

  /**
   * Close the run: set `endTime` and `finalSuccess`, then derive the total
   * time, per-step averages, cost per step and actual success rate.
   */
  finalize(success: boolean): void {
    this.endTime = Date.now();
    this.finalSuccess = success;

    if (this.startTime != null && this.endTime != null) {
      this.totalTimeSeconds = (this.endTime - this.startTime) / 1000;
    }

    if (this.completedSteps > 0) {
      this.averageVotesPerStep = this.subtaskMetrics.reduce((s, m) =>
        s + m.totalVotes, 0) /
        this.completedSteps;
      this.averageTimePerStepMs = this.subtaskMetrics.reduce((s, m) =>
        s + m.executionTimeMs, 0) /
        this.completedSteps;
      this.costPerStep = this.actualCostUsd / this.completedSteps;
    }

    if (this.totalSteps > 0) {
      this.actualSuccessRate = this.completedSteps / this.totalSteps;
    }

    if (this.minVotesForSingleStep === Infinity) {
      this.minVotesForSingleStep = 0;
    }
  }

  /** Multi-line human-readable summary of steps, samples, votes, cost, tokens, time and outcome. */
  summary(): string {
    const rfRate = this.totalSamples > 0
      ? (this.redFlaggedSamples / this.totalSamples) * 100
      : 0;
    return [
      `MDAP Execution Summary:`,
      `- Steps: ${this.completedSteps}/${this.totalSteps} completed (${this.failedSteps} failed)`,
      `- Samples: ${this.totalSamples} total, ${this.validSamples} valid, ${this.redFlaggedSamples} red-flagged (${
        rfRate.toFixed(1)
      }%)`,
      `- Avg votes/step: ${
        this.averageVotesPerStep.toFixed(1)
      } (min: ${this.minVotesForSingleStep}, max: ${this.maxVotesForSingleStep})`,
      `- Cost: $${this.actualCostUsd.toFixed(4)} ($${
        this.costPerStep.toFixed(6)
      }/step)`,
      `- Tokens: ${this.totalInputTokens} in, ${this.totalOutputTokens} out`,
      `- Time: ${this.totalTimeSeconds.toFixed(1)}s (${
        this.averageTimePerStepMs.toFixed(0)
      }ms/step)`,
      `- Success: ${this.finalSuccess ? "YES" : "NO"}`,
    ].join("\n");
  }

  /** Multi-line breakdown of red-flag reasons sorted by count with percentage of red-flagged samples. */
  redFlagAnalysis(): string {
    const keys = Object.keys(this.redFlagBreakdown);
    if (keys.length === 0) return "No red-flags encountered.";
    const sorted = keys.sort(
      (a, b) => this.redFlagBreakdown[b] - this.redFlagBreakdown[a],
    );
    let analysis = "Red-Flag Analysis:\n";
    for (const reason of sorted) {
      const count = this.redFlagBreakdown[reason];
      const pct = (count / Math.max(1, this.redFlaggedSamples)) * 100;
      analysis += `  - ${reason}: ${count} (${pct.toFixed(1)}%)\n`;
    }
    return analysis;
  }
}

// ---------------------------------------------------------------------------
// Tool intent helpers
// ---------------------------------------------------------------------------

/** Check if a tool name belongs to a tool category. */
export function toolCategoryContains(
  category: ToolCategory,
  toolName: string,
): boolean {
  if (typeof category === "object" && "custom" in category) {
    return toolName.startsWith(category.custom);
  }
  switch (category) {
    case "file_read":
      return ["read_file", "file_read", "get_file_contents"].includes(
        toolName,
      );
    case "file_write":
      return [
        "write_file",
        "edit_file",
        "delete_file",
        "create_directory",
        "file_write",
      ].includes(toolName);
    case "search":
      return ["search_files", "grep", "find_files", "glob", "file_search"]
        .includes(
          toolName,
        );
    case "semantic_search":
      return ["semantic_search", "query_codebase", "rag_search"].includes(
        toolName,
      );
    case "bash":
      return ["bash", "execute_command", "shell", "run_command"].includes(
        toolName,
      );
    case "git":
      return [
        "git",
        "git_status",
        "git_diff",
        "git_commit",
        "git_log",
      ].includes(toolName);
    case "web":
      return ["web_search", "fetch_url", "browse", "http_request"].includes(
        toolName,
      );
    case "agent_pool":
      return ["spawn_agent", "agent_pool", "create_agent"].includes(toolName);
    case "task_manager":
      return ["create_task", "update_task", "task_manager"].includes(toolName);
    case "mcp":
      return toolName.startsWith("mcp_") || toolName.startsWith("mcp__");
  }
}

/** Get all read-only tool categories. */
export function readOnlyCategories(): Set<ToolCategory> {
  return new Set<ToolCategory>(["file_read", "search", "semantic_search"]);
}

/** Get all side-effect tool categories. */
export function sideEffectCategories(): Set<ToolCategory> {
  return new Set<ToolCategory>([
    "file_write",
    "bash",
    "git",
    "web",
    "agent_pool",
    "task_manager",
  ]);
}

/** Format a tool schema for inclusion in prompts. */
export function toolSchemaToPrompt(schema: ToolSchema): string {
  let result = `- **${schema.name}**: ${schema.description}\n`;
  const params = Object.entries(schema.parameters);
  if (params.length > 0) {
    result += "  Parameters:\n";
    for (const [name, desc] of params) {
      const required = schema.required.includes(name) ? " (required)" : "";
      result += `    - ${name}${required}: ${desc}\n`;
    }
  }
  return result;
}

/** Parse tool intent from a microagent's text response. */
export function parseToolIntent(
  subtaskId: string,
  responseText: string,
):
  | { kind: "no_intent"; output: SubtaskOutput }
  | { kind: "with_intent"; output: SubtaskOutput; toolIntent: ToolIntent }
  | { kind: "parse_error"; message: string } {
  // Try to find JSON block with tool_intent
  const jsonBlock = extractJsonCodeBlock(responseText);
  if (jsonBlock) {
    try {
      const value = JSON.parse(jsonBlock);
      const intent = value.tool_intent ?? value;
      if (intent.tool_name) {
        const outputText = removeJsonBlock(responseText).trim();
        return {
          kind: "with_intent",
          output: {
            subtaskId,
            value: { text: outputText, awaiting_tool: true },
          },
          toolIntent: {
            toolName: intent.tool_name,
            arguments: intent.arguments ?? {},
            rationale: intent.rationale,
          },
        };
      }
    } catch {
      // fall through
    }
  }

  // Try inline JSON
  for (const line of responseText.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
      try {
        const value = JSON.parse(trimmed);
        const intent = value.tool_intent ?? value;
        if (intent.tool_name) {
          return {
            kind: "with_intent",
            output: {
              subtaskId,
              value: { text: responseText, awaiting_tool: true },
            },
            toolIntent: {
              toolName: intent.tool_name,
              arguments: intent.arguments ?? {},
              rationale: intent.rationale,
            },
          };
        }
      } catch {
        // ignore
      }
    }
  }

  return {
    kind: "no_intent",
    output: { subtaskId, value: { text: responseText } },
  };
}

function extractJsonCodeBlock(text: string): string | null {
  for (const marker of ["```json", "```JSON"]) {
    const start = text.indexOf(marker);
    if (start === -1) continue;
    const contentStart = start + marker.length;
    const end = text.indexOf("```", contentStart);
    if (end === -1) continue;
    return text.slice(contentStart, end).trim();
  }
  return null;
}

function removeJsonBlock(text: string): string {
  let result = text;
  for (const marker of ["```json", "```JSON"]) {
    const start = result.indexOf(marker);
    if (start === -1) continue;
    const contentStart = start + marker.length;
    const end = result.indexOf("```", contentStart);
    if (end === -1) continue;
    result = result.slice(0, start) + result.slice(end + 3);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Borda count standalone
// ---------------------------------------------------------------------------

/** Borda count voting - ranks candidates by weighted confidence scores. */
export function bordaCountWinner<T>(
  votes: { key: string; value: T; confidence: number }[],
): { key: string; value: T; score: number } | null {
  if (votes.length === 0) return null;
  const scores = new Map<string, { score: number; value: T }>();
  for (const v of votes) {
    const entry = scores.get(v.key) ?? { score: 0, value: v.value };
    entry.score += v.confidence;
    scores.set(v.key, entry);
  }
  let best: { key: string; value: T; score: number } | null = null;
  for (const [key, { score, value }] of scores) {
    if (!best || score > best.score) {
      best = { key, value, score };
    }
  }
  return best;
}
