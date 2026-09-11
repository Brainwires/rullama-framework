/**
 * @module @rullama/inference
 *
 * LLM-driven agent workhorses for the rullama.
 *
 * Extracted from `@rullama/agents` in v0.11.0 to mirror Rust's
 * `rullama-inference` crate. Provides:
 *
 * - **AgentRuntime / runAgentLoop** — generic execution loop
 * - **TaskAgent** — concrete agent with provider + tool loop
 * - **AgentContext** — environment bundle
 * - **JudgeAgent / PlannerAgent / ValidatorAgent** — LLM-driven helpers
 * - **CycleOrchestrator** — Plan → Work → Judge loop
 * - **PlanExecutorAgent** — plan execution
 * - **ValidationLoop** — quality checks
 * - **AgentLifecycleHooks** — hook interface for telemetry/observability
 * - **AgentRole** — least-privilege tool restriction by role
 * - **system_prompts** — canonical prompt registry
 *
 * Coordination primitives (communication, locks, task manager/queue,
 * patterns) stay in `@rullama/agents` / `@rullama/agent`.
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
