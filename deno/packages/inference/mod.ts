/**
 * @module @rullama/inference
 *
 * LLM-driven agent workhorses for the rullama.
 *
 * Extracted from `@rullama/agents` in v0.11.0 to mirror Rust's
 * `rullama-inference` crate. Provides:
 *
 * - **AgentRuntime / runAgentLoop** — generic execution loop
 * - **TaskAgent / spawnTaskAgent** — concrete agent with provider + tool loop
 * - **AgentContext** — environment bundle; wraps the tool executor in
 *   `@rullama/tool-runtime`'s enforcing executor by default
 *   (`enforcement: false` opts out)
 * - **AgentPool** — bounded pool of concurrent `TaskAgent`s
 * - **ValidatorAgent** / **runValidation** — read-only quality gates
 * - **PlanExecutorAgent** — plan execution with approval modes
 * - **Judge / Planner helpers** — prompt builders and parsers
 *   (`judgeAgentPrompt`, `parseVerdict`, `plannerAgentPrompt`,
 *   `parsePlannerOutput`, `validateTaskGraph`) plus their config types. There
 *   are no `JudgeAgent` / `PlannerAgent` classes; drive a `TaskAgent` with the
 *   prompt and parse its output.
 * - **Cycle orchestration** — `CycleOrchestratorConfig` / `CycleRecord` /
 *   `CycleOrchestratorResult` types only; the Plan → Work → Judge loop itself
 *   is not ported.
 * - **AgentLifecycleHooks** — hook interface for telemetry/observability
 * - **AgentRole** — least-privilege tool restriction by role
 * - **system_prompts** — canonical prompt registry
 *
 * Coordination primitives (communication, locks, task manager/queue,
 * patterns) stay in `@rullama/agent`.
 */

export * from "./runtime.ts";
export * from "./context.ts";
export * from "./hooks.ts";
export * from "./task_agent.ts";
export * from "./judge_agent.ts";
export * from "./planner_agent.ts";
export * from "./validator_agent.ts";
export * from "./plan_executor.ts";
export * from "./cycle_orchestrator.ts";
export * from "./validation_loop.ts";
export * from "./roles.ts";
export * from "./agent_pool.ts";

// System prompt registry. `judgeAgentPrompt` / `plannerAgentPrompt` have one
// implementation (system_prompts/agents.ts); the `canonical*` aliases are kept
// for 0.12 compatibility and are removed in 0.13.
export {
  type AgentPromptKind,
  buildAgentPrompt,
  /** @deprecated Use `judgeAgentPrompt`. */
  judgeAgentPrompt as canonicalJudgeAgentPrompt,
  mdapMicroagentPrompt,
  /** @deprecated Use `plannerAgentPrompt`. */
  plannerAgentPrompt as canonicalPlannerAgentPrompt,
  reasoningAgentPrompt,
  simpleAgentPrompt,
} from "./system_prompts/mod.ts";
