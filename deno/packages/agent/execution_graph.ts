/**
 * Execution DAG and telemetry for TaskAgent runs.
 *
 * Provides {@link ExecutionGraph} (one node per provider-call iteration with
 * tool call records) and {@link RunTelemetry} (aggregate summary derived from
 * the graph at run completion).
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** One tool call within a single iteration step. */
export interface ToolCallRecord {
  /** Unique identifier for this tool use invocation. */
  toolUseId: string;
  /** Name of the tool that was called. */
  toolName: string;
  /** Whether the tool call resulted in an error. */
  isError: boolean;
  /** When the tool call was executed (ISO string). */
  executedAt: string;
}

/** One provider-call iteration in the execute() loop. */
export interface StepNode {
  /** Iteration number within the execution loop. */
  iteration: number;
  /** When this step started (ISO string). */
  startedAt: string;
  /** When this step ended (ISO string). */
  endedAt: string;
  /** Prompt tokens for this call. */
  promptTokens: number;
  /** Completion tokens for this call. */
  completionTokens: number;
  /** Tool calls made during this step. */
  toolCalls: ToolCallRecord[];
  /** Reason the provider stopped generating. */
  finishReason: string | null;
}

// ---------------------------------------------------------------------------
// ExecutionGraph
// ---------------------------------------------------------------------------

/** Full execution trace for one TaskAgent run. */
export class ExecutionGraph {
  /** SHA-256 of (system prompt + sorted tool name bytes), hex-encoded. */
  promptHash: string;
  /** When the run started (ISO string). */
  runStartedAt: string;
  /** One StepNode per provider call iteration. */
  steps: StepNode[] = [];
  /** Flat ordered list of tool names across all steps. */
  toolSequence: string[] = [];

  /**
   * Create an empty graph for one run.
   * @param promptHash Hex SHA-256 identifying the system prompt + tool set.
   * @param runStartedAt ISO timestamp of the run start; defaults to now.
   */
  constructor(promptHash: string, runStartedAt?: string) {
    this.promptHash = promptHash;
    this.runStartedAt = runStartedAt ?? new Date().toISOString();
  }

  /** Start a new step; returns its index for later finalization. */
  pushStep(iteration: number, startedAt?: string): number {
    const idx = this.steps.length;
    const ts = startedAt ?? new Date().toISOString();
    this.steps.push({
      iteration,
      startedAt: ts,
      endedAt: ts,
      promptTokens: 0,
      completionTokens: 0,
      toolCalls: [],
      finishReason: null,
    });
    return idx;
  }

  /** Fill in token counts and finish_reason after the provider call returns. */
  finalizeStep(
    stepIdx: number,
    endedAt: string,
    promptTokens: number,
    completionTokens: number,
    finishReason: string | null,
  ): void {
    const step = this.steps[stepIdx];
    if (!step) return;
    step.endedAt = endedAt;
    step.promptTokens = promptTokens;
    step.completionTokens = completionTokens;
    step.finishReason = finishReason;
  }

  /** Record a tool call and append its name to the flat sequence. */
  recordToolCall(stepIdx: number, record: ToolCallRecord): void {
    this.toolSequence.push(record.toolName);
    const step = this.steps[stepIdx];
    if (step) step.toolCalls.push(record);
  }
}

// ---------------------------------------------------------------------------
// RunTelemetry
// ---------------------------------------------------------------------------

/** Structured telemetry summary for a completed run. */
export interface RunTelemetry {
  /** Hash of the system prompt and tool registry. */
  promptHash: string;
  /** When the run started (ISO string). */
  runStartedAt: string;
  /** When the run ended (ISO string). */
  runEndedAt: string;
  /** Total run duration in milliseconds. */
  durationMs: number;
  /** Number of provider call iterations. */
  totalIterations: number;
  /** Total number of tool calls across all iterations. */
  totalToolCalls: number;
  /** Number of tool calls that returned errors. */
  toolErrorCount: number;
  /** Unique tool names, deduped in first-use order. */
  toolsUsed: string[];
  /** Total prompt tokens consumed. */
  totalPromptTokens: number;
  /** Total completion tokens consumed. */
  totalCompletionTokens: number;
  /** Total estimated cost in USD. */
  totalCostUsd: number;
  /** Whether the run completed successfully. */
  success: boolean;
}

/** Build a telemetry record from a completed ExecutionGraph. */
export function telemetryFromGraph(
  graph: ExecutionGraph,
  runEndedAt: string,
  success: boolean,
  totalCostUsd: number,
): RunTelemetry {
  const startMs = new Date(graph.runStartedAt).getTime();
  const endMs = new Date(runEndedAt).getTime();
  const durationMs = Math.max(0, endMs - startMs);

  let totalToolCalls = 0;
  let toolErrorCount = 0;
  let totalPromptTokens = 0;
  let totalCompletionTokens = 0;

  for (const step of graph.steps) {
    totalToolCalls += step.toolCalls.length;
    for (const tc of step.toolCalls) {
      if (tc.isError) toolErrorCount++;
    }
    totalPromptTokens += step.promptTokens;
    totalCompletionTokens += step.completionTokens;
  }

  const seen = new Set<string>();
  const toolsUsed: string[] = [];
  for (const name of graph.toolSequence) {
    if (!seen.has(name)) {
      seen.add(name);
      toolsUsed.push(name);
    }
  }

  return {
    promptHash: graph.promptHash,
    runStartedAt: graph.runStartedAt,
    runEndedAt,
    durationMs,
    totalIterations: graph.steps.length,
    totalToolCalls,
    toolErrorCount,
    toolsUsed,
    totalPromptTokens,
    totalCompletionTokens,
    totalCostUsd,
    success,
  };
}
