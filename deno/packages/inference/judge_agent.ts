/**
 * Judge Agent - LLM-powered cycle evaluator.
 *
 * Wraps a TaskAgent with a judge-specific system prompt to evaluate
 * the results of a Plan->Work cycle and produce a {@link JudgeVerdict}
 * that determines what happens next: complete, continue, fresh restart,
 * or abort.
 *
 * @module
 */

import type { DynamicTaskSpec } from "./planner_agent.ts";

// ---------------------------------------------------------------------------
// Verdict types
// ---------------------------------------------------------------------------

/** The judge's decision after evaluating a cycle. */
export type JudgeVerdict =
  | { verdict: "complete"; summary: string }
  | {
    verdict: "continue";
    summary: string;
    additionalTasks: DynamicTaskSpec[];
    retryTasks: string[];
    hints: string[];
  }
  | {
    verdict: "fresh_restart";
    reason: string;
    hints: string[];
    summary: string;
  }
  | { verdict: "abort"; reason: string; summary: string };

/** Get the verdict type string. */
export function verdictType(v: JudgeVerdict): string {
  return v.verdict;
}

/** Get hints from a verdict (empty array for complete/abort). */
export function verdictHints(v: JudgeVerdict): string[] {
  if (v.verdict === "continue" || v.verdict === "fresh_restart") {
    return v.hints;
  }
  return [];
}

// ---------------------------------------------------------------------------
// Merge status
// ---------------------------------------------------------------------------

/** Merge status for a worker's branch. */
export type MergeStatus =
  | { kind: "merged" }
  | { kind: "conflict_resolved" }
  | { kind: "conflict_failed"; message: string }
  | { kind: "not_attempted" };

/** Format merge status as a display string. */
export function formatMergeStatus(status: MergeStatus): string {
  switch (status.kind) {
    case "merged":
      return "merged";
    case "conflict_resolved":
      return "conflict_resolved";
    case "conflict_failed":
      return `conflict_failed: ${status.message}`;
    case "not_attempted":
      return "not_attempted";
  }
}

// ---------------------------------------------------------------------------
// Worker result
// ---------------------------------------------------------------------------

/** Result from a single worker in the cycle. */
export interface WorkerResult {
  /** ID of the task the worker executed. */
  taskId: string;
  /** Description of the task. */
  taskDescription: string;
  /** Whether the worker reported success. */
  success: boolean;
  /** Worker's summary of what it did. */
  summary: string;
  /** Iterations the worker used. */
  iterations: number;
  /** Branch the worker committed to. */
  branchName: string;
  /** Result of merging the worker's branch. */
  mergeStatus: MergeStatus;
}

// ---------------------------------------------------------------------------
// Judge context
// ---------------------------------------------------------------------------

/** Context provided to the judge for evaluation. */
export interface JudgeContext {
  /** The goal the whole cycle is working toward. */
  originalGoal: string;
  /** 1-based number of the cycle being judged. */
  cycleNumber: number;
  /** Results of every worker in this cycle. */
  workerResults: WorkerResult[];
  /** Why the planner chose this cycle's tasks. */
  plannerRationale: string;
  /** Verdicts from earlier cycles. */
  previousVerdicts: JudgeVerdict[];
}

// ---------------------------------------------------------------------------
// Judge agent config
// ---------------------------------------------------------------------------

/** Configuration for the judge agent. */
export interface JudgeAgentConfig {
  /** Maximum iterations the judge agent may use. */
  maxIterations: number;
  /** Whether the judge may read files to verify claims. */
  inspectFiles: boolean;
  /** Whether the judge may inspect git diffs. */
  inspectDiffs: boolean;
  /** Sampling temperature for the judge model. */
  temperature: number;
  /** Maximum tokens per judge response. */
  maxTokens: number;
}

/** Default judge agent config. */
export function defaultJudgeAgentConfig(): JudgeAgentConfig {
  return {
    maxIterations: 15,
    inspectFiles: true,
    inspectDiffs: true,
    temperature: 0.3,
    maxTokens: 4096,
  };
}

// ---------------------------------------------------------------------------
// System prompt generation
// ---------------------------------------------------------------------------

// The system prompt lives in system_prompts/agents.ts (the canonical wording);
// re-exported here so `import { judgeAgentPrompt } from "./judge_agent.ts"` keeps working.
export { judgeAgentPrompt } from "./system_prompts/agents.ts";

// ---------------------------------------------------------------------------
// Task description builder
// ---------------------------------------------------------------------------

/** Build the task description that gives the judge full context. */
export function buildJudgeTaskDescription(ctx: JudgeContext): string {
  let desc = `# Evaluate Cycle ${ctx.cycleNumber} Results\n\n`;
  desc += `## Original Goal\n${ctx.originalGoal}\n\n`;
  desc += `## Planner Rationale\n${ctx.plannerRationale}\n\n`;
  desc += `## Worker Results\n\n`;

  for (let i = 0; i < ctx.workerResults.length; i++) {
    const wr = ctx.workerResults[i];
    desc += `### Worker ${i + 1} (task: ${wr.taskId})\n`;
    desc += `- **Task**: ${wr.taskDescription}\n`;
    desc += `- **Success**: ${wr.success}\n`;
    desc += `- **Summary**: ${wr.summary}\n`;
    desc += `- **Branch**: ${wr.branchName}\n`;
    desc += `- **Merge**: ${formatMergeStatus(wr.mergeStatus)}\n`;
    desc += `- **Iterations**: ${wr.iterations}\n\n`;
  }

  if (ctx.previousVerdicts.length > 0) {
    desc += `## Previous Verdicts\n\n`;
    for (let i = 0; i < ctx.previousVerdicts.length; i++) {
      desc += `- Cycle ${i}: ${verdictType(ctx.previousVerdicts[i])}\n`;
    }
    desc += "\n";
  }

  desc += "## Your Task\n\n" +
    "Evaluate the above results against the original goal. " +
    "Output your verdict as a JSON block. " +
    "If you need to inspect files or diffs for verification, use the available tools first.";

  return desc;
}

// ---------------------------------------------------------------------------
// Verdict parsing
// ---------------------------------------------------------------------------

/** Extract a JSON block from text (searches for ```json fences, then raw JSON). */
export function extractJsonBlock(text: string): string | null {
  // Try ```json ... ``` fences
  const jsonFenceStart = text.indexOf("```json");
  if (jsonFenceStart !== -1) {
    const contentStart = jsonFenceStart + "```json".length;
    const end = text.indexOf("```", contentStart);
    if (end !== -1) {
      return text.slice(contentStart, end).trim();
    }
  }

  // Try ``` ... ``` fences
  const fenceStart = text.indexOf("```");
  if (fenceStart !== -1) {
    const contentStart = fenceStart + "```".length;
    const lineEnd = text.indexOf("\n", contentStart);
    if (lineEnd !== -1) {
      const actualStart = lineEnd + 1;
      const end = text.indexOf("```", actualStart);
      if (end !== -1) {
        const candidate = text.slice(actualStart, end).trim();
        if (candidate.startsWith("{")) return candidate;
      }
    }
  }

  // Try raw JSON
  const braceStart = text.indexOf("{");
  if (braceStart !== -1) {
    let depth = 0;
    let end = braceStart;
    for (let i = braceStart; i < text.length; i++) {
      if (text[i] === "{") depth++;
      else if (text[i] === "}") {
        depth--;
        if (depth === 0) {
          end = i + 1;
          break;
        }
      }
    }
    if (depth === 0 && end > braceStart) {
      return text.slice(braceStart, end);
    }
  }

  return null;
}

/** Parse a judge verdict from text output. */
export function parseVerdict(text: string): JudgeVerdict {
  const jsonStr = extractJsonBlock(text);
  if (!jsonStr) throw new Error("No JSON block found in judge output");

  const raw = JSON.parse(jsonStr);

  switch (raw.verdict) {
    case "complete":
      return { verdict: "complete", summary: raw.summary ?? "" };
    case "continue":
      return {
        verdict: "continue",
        summary: raw.summary ?? "",
        additionalTasks: raw.additional_tasks ?? [],
        retryTasks: raw.retry_tasks ?? [],
        hints: raw.hints ?? [],
      };
    case "fresh_restart":
      return {
        verdict: "fresh_restart",
        reason: raw.reason ?? "",
        hints: raw.hints ?? [],
        summary: raw.summary ?? "",
      };
    case "abort":
      return {
        verdict: "abort",
        reason: raw.reason ?? "",
        summary: raw.summary ?? "",
      };
    default:
      throw new Error(`Unknown verdict type: ${raw.verdict}`);
  }
}
