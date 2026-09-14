// deno-lint-ignore-file no-explicit-any

function nowTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

/** Status of a plan.
 * Equivalent to Rust's `PlanStatus` in rullama-core. */
export type PlanStatus =
  | "draft"
  | "active"
  | "paused"
  | "completed"
  | "abandoned";

/** Parse a PlanStatus from a string. */
export function parsePlanStatus(s: string): PlanStatus | undefined {
  const lower = s.toLowerCase();
  if (["draft", "active", "paused", "completed", "abandoned"].includes(lower)) {
    return lower as PlanStatus;
  }
  return undefined;
}

/** Metadata for a persisted execution plan.
 * Equivalent to Rust's `PlanMetadata` in rullama-core. */
export class PlanMetadata {
  /** Random UUID identifying the plan. */
  plan_id: string;
  /** Conversation the plan belongs to. */
  conversation_id: string;
  /** First line of the task description, truncated to 50 characters. */
  title: string;
  /** The task the plan was written for. */
  task_description: string;
  /** The plan text itself (markdown). */
  plan_content: string;
  /** Model that generated the plan, if recorded. */
  model_id?: string;
  /** Current lifecycle status. */
  status: PlanStatus;
  /** True once the plan has been run (see {@link PlanMetadata.markExecuted}). */
  executed: boolean;
  /** Agent iterations consumed while executing the plan. */
  iterations_used: number;
  /** Unix timestamp (seconds) when the plan was created. */
  created_at: number;
  /** Unix timestamp (seconds) of the last mutation. */
  updated_at: number;
  /** Path the plan was exported to, once {@link PlanMetadata.setFilePath} has been called. */
  file_path?: string;
  /** Embedding vector of the plan, for similarity lookup. */
  embedding?: number[];
  /** ID of the plan this one branched from; undefined for a root plan. */
  parent_plan_id?: string;
  /** IDs of plans branched from this one. */
  child_plan_ids: string[];
  /** Name given to this branch when it was created from its parent. */
  branch_name?: string;
  /** True once the branch has been merged back into its parent. */
  merged: boolean;
  /** Nesting depth: 0 for a root plan, parent depth + 1 for a branch. */
  depth: number;

  /**
   * Create a draft plan with a fresh UUID; the title is derived from the
   * first line of `taskDescription`.
   */
  constructor(
    conversationId: string,
    taskDescription: string,
    planContent: string,
  ) {
    const now = nowTimestamp();
    this.plan_id = crypto.randomUUID();
    this.conversation_id = conversationId;
    this.title = (taskDescription.split("\n")[0] ?? taskDescription).slice(
      0,
      50,
    );
    this.task_description = taskDescription;
    this.plan_content = planContent;
    this.status = "draft";
    this.executed = false;
    this.iterations_used = 0;
    this.created_at = now;
    this.updated_at = now;
    this.child_plan_ids = [];
    this.merged = false;
    this.depth = 0;
  }

  /** Create a branch (sub-plan) from this plan. */
  createBranch(
    branchName: string,
    taskDescription: string,
    planContent: string,
  ): PlanMetadata {
    const branch = new PlanMetadata(
      this.conversation_id,
      taskDescription,
      planContent,
    );
    branch.parent_plan_id = this.plan_id;
    branch.branch_name = branchName;
    branch.depth = this.depth + 1;
    return branch;
  }

  /** Add a child plan ID. */
  addChild(childId: string): void {
    if (!this.child_plan_ids.includes(childId)) {
      this.child_plan_ids.push(childId);
      this.updated_at = nowTimestamp();
    }
  }

  /** Mark as merged. */
  markMerged(): void {
    this.merged = true;
    this.status = "completed";
    this.updated_at = nowTimestamp();
  }

  /** Check if this is a root plan. */
  isRoot(): boolean {
    return this.parent_plan_id === undefined;
  }

  /** Check if this plan has children. */
  hasChildren(): boolean {
    return this.child_plan_ids.length > 0;
  }

  /** Set the model used (builder). */
  withModel(modelId: string): this {
    this.model_id = modelId;
    return this;
  }

  /** Set iterations used (builder). */
  withIterations(iterations: number): this {
    this.iterations_used = iterations;
    return this;
  }

  /** Mark as executed. */
  markExecuted(): void {
    this.executed = true;
    this.status = "completed";
    this.updated_at = nowTimestamp();
  }

  /** Update status. */
  setStatus(status: PlanStatus): void {
    this.status = status;
    this.updated_at = nowTimestamp();
  }

  /** Set file path after export. */
  setFilePath(path: string): void {
    this.file_path = path;
    this.updated_at = nowTimestamp();
  }

  /** Generate markdown export with YAML frontmatter. */
  toMarkdown(): string {
    const created = new Date(this.created_at * 1000).toISOString().replace(
      /\.\d{3}Z$/,
      "Z",
    );
    const model = this.model_id ?? "unknown";
    const escapedTitle = this.title.replace(/"/g, '\\"');
    return `---
plan_id: ${this.plan_id}
conversation_id: ${this.conversation_id}
title: "${escapedTitle}"
status: ${this.status}
executed: ${this.executed}
iterations: ${this.iterations_used}
created_at: ${created}
model: ${model}
---

# Execution Plan: ${this.title}

## Original Task

${this.task_description}

## Plan

${this.plan_content}

---
*Generated by rullama*
`;
  }
}

/** A single step in a serializable pre-execution plan.
 * Equivalent to Rust's `PlanStep` in rullama-core. */
export interface PlanStep {
  /** 1-based position of the step within the plan. */
  step_number: number;
  /** What the step does. */
  description: string;
  /** Name of the tool the step is expected to use, if the model suggested one. */
  tool_hint?: string;
  /** Model's estimate of the tokens the step will consume. */
  estimated_tokens: number;
}

/** Budget constraints for a serializable plan.
 * Equivalent to Rust's `PlanBudget` in rullama-core. */
export class PlanBudget {
  /** Maximum number of steps a plan may have; unlimited when undefined. */
  max_steps?: number;
  /** Maximum total estimated tokens; unlimited when undefined. */
  max_estimated_tokens?: number;
  /** Maximum estimated cost in USD; unlimited when undefined. */
  max_estimated_cost_usd?: number;
  /** USD per token used to turn the token estimate into a cost (default 0.000003). */
  cost_per_token: number;

  /** Create an unlimited budget with the default per-token cost. */
  constructor() {
    this.cost_per_token = 0.000003;
  }

  /** Set the maximum allowed step count (builder). */
  withMaxSteps(max: number): this {
    this.max_steps = max;
    return this;
  }

  /** Set the maximum allowed estimated token count (builder). */
  withMaxTokens(max: number): this {
    this.max_estimated_tokens = max;
    return this;
  }

  /** Set the maximum allowed estimated cost in USD (builder). */
  withMaxCostUsd(max: number): this {
    this.max_estimated_cost_usd = max;
    return this;
  }

  /** Check a plan against this budget. Returns undefined if OK, or error string. */
  check(plan: SerializablePlan): string | undefined {
    const stepCount = plan.steps.length;
    const totalTokens = plan.totalEstimatedTokens();
    const totalCost = totalTokens * this.cost_per_token;

    if (this.max_steps !== undefined && stepCount > this.max_steps) {
      return `plan has ${stepCount} steps but limit is ${this.max_steps}`;
    }
    if (
      this.max_estimated_tokens !== undefined &&
      totalTokens > this.max_estimated_tokens
    ) {
      return `plan estimates ${totalTokens} tokens but limit is ${this.max_estimated_tokens}`;
    }
    if (
      this.max_estimated_cost_usd !== undefined &&
      totalCost > this.max_estimated_cost_usd
    ) {
      return `plan estimates $${totalCost.toFixed(6)} USD but limit is $${
        this.max_estimated_cost_usd.toFixed(6)
      }`;
    }
    return undefined;
  }
}

/** A serializable execution plan.
 * Equivalent to Rust's `SerializablePlan` in rullama-core. */
export class SerializablePlan {
  /** Random UUID identifying the plan. */
  plan_id: string;
  /** The task the plan was written for. */
  task_description: string;
  /** Ordered steps of the plan. */
  steps: PlanStep[];
  /** Unix timestamp (seconds) when the plan was created. */
  created_at: number;

  /** Create a plan for `taskDescription` from already-parsed steps. */
  constructor(taskDescription: string, steps: PlanStep[]) {
    this.plan_id = crypto.randomUUID();
    this.task_description = taskDescription;
    this.steps = steps;
    this.created_at = nowTimestamp();
  }

  /** Sum of all estimated_tokens across steps. */
  totalEstimatedTokens(): number {
    return this.steps.reduce((sum, s) => sum + s.estimated_tokens, 0);
  }

  /** Number of steps in this plan. */
  stepCount(): number {
    return this.steps.length;
  }

  /** Parse a plan from model-generated text that contains an embedded JSON object with a "steps" array. */
  static parseFromText(
    taskDescription: string,
    text: string,
  ): SerializablePlan | undefined {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start === -1 || end === -1 || start > end) return undefined;

    let value: any;
    try {
      value = JSON.parse(text.slice(start, end + 1));
    } catch {
      return undefined;
    }

    const stepsArray = value?.steps;
    if (!Array.isArray(stepsArray)) return undefined;

    const steps: PlanStep[] = [];
    for (let i = 0; i < stepsArray.length; i++) {
      const step = stepsArray[i];
      const description = step?.description;
      if (typeof description !== "string") continue;

      const estimatedTokens = step?.estimated_tokens ?? step?.tokens ?? 500;
      const toolHint = step?.tool ?? step?.tool_hint;

      steps.push({
        step_number: i + 1,
        description,
        tool_hint: typeof toolHint === "string" ? toolHint : undefined,
        estimated_tokens: typeof estimatedTokens === "number"
          ? estimatedTokens
          : 500,
      });
    }

    if (steps.length === 0) return undefined;
    return new SerializablePlan(taskDescription, steps);
  }
}
