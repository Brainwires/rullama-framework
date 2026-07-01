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

// System prompt registry. `judgeAgentPrompt` and `plannerAgentPrompt` exist in
// two places (judge_agent.ts / planner_agent.ts with drifted wording vs the
// canonical system_prompts/agents.ts version). We export the canonical
// versions aliased — consumers wanting strict Rust↔Deno parity should use
// `canonicalJudgeAgentPrompt` / `canonicalPlannerAgentPrompt`.
export {
  type AgentPromptKind,
  buildAgentPrompt,
  judgeAgentPrompt as canonicalJudgeAgentPrompt,
  mdapMicroagentPrompt,
  plannerAgentPrompt as canonicalPlannerAgentPrompt,
  reasoningAgentPrompt,
  simpleAgentPrompt,
} from "./system_prompts/mod.ts";
