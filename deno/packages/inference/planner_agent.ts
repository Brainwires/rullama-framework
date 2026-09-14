/**
 * Planner Agent - LLM-powered dynamic task planner.
 *
 * Wraps a TaskAgent with a planner-specific system prompt. It explores
 * the codebase using read-only tools and outputs structured JSON
 * describing tasks for worker agents to execute.
 *
 * The planner never directly mutates the task graph -- it produces a
 * {@link PlannerOutput} that the CycleOrchestrator interprets.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Priority level for dynamically created tasks. */
export type DynamicTaskPriority = "urgent" | "high" | "normal" | "low";

/** A task specification created dynamically by the planner at runtime. */
export interface DynamicTaskSpec {
  /** Unique identifier. */
  id: string;
  /** Clear description of what the worker should do. */
  description: string;
  /** File paths the task is expected to touch. */
  filesInvolved: string[];
  /** IDs of other specs this task depends on. */
  dependsOn: string[];
  /** Task priority. */
  priority: DynamicTaskPriority;
  /** Estimated iterations the worker will need. */
  estimatedIterations: number | null;
}

/** Request to spawn a sub-planner for a specific focus area. */
export interface SubPlannerRequest {
  /** Area of the codebase to focus on. */
  focusArea: string;
  /** Additional context for the sub-planner. */
  context: string;
  /** Maximum recursion depth remaining. */
  maxDepth: number;
}

/** Output produced by a planner agent run. */
export interface PlannerOutput {
  /** Tasks to execute in this cycle. */
  tasks: DynamicTaskSpec[];
  /** Optional sub-planners to spawn for deeper analysis. */
  subPlanners: SubPlannerRequest[];
  /** Brief explanation of the overall plan. */
  rationale: string;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/** Configuration for the planner agent. */
export interface PlannerAgentConfig {
  /** LLM call budget for planning. */
  maxIterations: number;
  /** Maximum number of tasks per cycle. */
  maxTasks: number;
  /** Maximum number of sub-planners to spawn. */
  maxSubPlanners: number;
  /** Maximum recursion depth for sub-planners. */
  planningDepth: number;
  /** Temperature for the planning LLM call. */
  temperature: number;
  /** Max tokens per LLM response. */
  maxTokens: number;
}

/** Default planner agent config. */
export function defaultPlannerAgentConfig(): PlannerAgentConfig {
  return {
    maxIterations: 20,
    maxTasks: 15,
    maxSubPlanners: 3,
    planningDepth: 2,
    temperature: 0.7,
    maxTokens: 4096,
  };
}

// ---------------------------------------------------------------------------
// System prompt generation
// ---------------------------------------------------------------------------

// The system prompt lives in system_prompts/agents.ts (the canonical wording);
// re-exported here so `import { plannerAgentPrompt } from "./planner_agent.ts"` keeps working.
export { plannerAgentPrompt } from "./system_prompts/agents.ts";

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/**
 * Parse planner output from text.
 *
 * Extracts JSON from markdown code fences or raw JSON, applies limits
 * from config, assigns IDs to tasks without them, and validates the
 * dependency graph.
 */
export function parsePlannerOutput(
  text: string,
  config: PlannerAgentConfig,
): PlannerOutput {
  // Extract JSON block (reuses same logic as judge_agent)
  const jsonStr = extractJsonBlockForPlanner(text);
  if (!jsonStr) throw new Error("No JSON block found in planner output");

  const raw = JSON.parse(jsonStr);

  const tasks: DynamicTaskSpec[] = (raw.tasks ?? [])
    .slice(0, config.maxTasks)
    .map(
      // deno-lint-ignore no-explicit-any
      (t: any): DynamicTaskSpec => ({
        id: t.id || crypto.randomUUID(),
        description: t.description ?? "",
        filesInvolved: t.files_involved ?? [],
        dependsOn: t.depends_on ?? [],
        priority: t.priority ?? "normal",
        estimatedIterations: t.estimated_iterations ?? null,
      }),
    );

  const subPlanners: SubPlannerRequest[] = (raw.sub_planners ?? [])
    .slice(0, config.maxSubPlanners)
    .map(
      // deno-lint-ignore no-explicit-any
      (s: any): SubPlannerRequest => ({
        focusArea: s.focus_area ?? "",
        context: s.context ?? "",
        maxDepth: s.max_depth ?? 1,
      }),
    );

  const output: PlannerOutput = {
    tasks,
    subPlanners,
    rationale: raw.rationale ?? "",
  };

  // Validate: no circular dependencies
  validateTaskGraph(output.tasks);

  return output;
}

// ---------------------------------------------------------------------------
// Task graph validation
// ---------------------------------------------------------------------------

/** Validate that a set of task specs has no circular dependencies. */
export function validateTaskGraph(tasks: DynamicTaskSpec[]): void {
  const idSet = new Set(tasks.map((t) => t.id));

  // Kahn's algorithm for cycle detection
  const inDegree = new Map<string, number>();
  for (const task of tasks) {
    const count = task.dependsOn.filter((d) => idSet.has(d)).length;
    inDegree.set(task.id, count);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  let visited = 0;
  while (queue.length > 0) {
    const node = queue.shift()!;
    visited++;
    for (const task of tasks) {
      if (task.dependsOn.includes(node) && idSet.has(task.id)) {
        const deg = inDegree.get(task.id)!;
        inDegree.set(task.id, deg - 1);
        if (deg - 1 === 0) queue.push(task.id);
      }
    }
  }

  if (visited < tasks.length) {
    throw new Error("Circular dependency detected in planner task graph");
  }
}

// ---------------------------------------------------------------------------
// JSON extraction helper
// ---------------------------------------------------------------------------

function extractJsonBlockForPlanner(text: string): string | null {
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
