# @rullama/inference

LLM-driven agent workhorses: the generic agent loop (`runAgentLoop` over an
`AgentRuntime`), `TaskAgent` / `spawnTaskAgent`, `AgentContext`, `AgentPool`,
`ValidatorAgent` + `runValidation`, `PlanExecutorAgent`, the judge / planner
prompt builders and parsers, `AgentRole` tool restriction, lifecycle hooks and
the system-prompt registry.

Extracted from the old `@rullama/agents` package in v0.11.0 to mirror Rust's
`rullama-inference` crate. The coordination primitives (`CommunicationHub`,
`FileLockManager`, `TaskManager`, …) live in `@rullama/agent`.

**Not ported (Rust-only):** `ChatAgent`, the `JudgeAgent` / `PlannerAgent`
classes and the `CycleOrchestrator` loop. Deno ships the judge / planner prompts
and parsers (`judgeAgentPrompt`, `parseVerdict`, `plannerAgentPrompt`,
`parsePlannerOutput`, `validateTaskGraph`) and the cycle-orchestrator
configuration types only.

## Install

```sh
deno add jsr:@rullama/inference jsr:@rullama/agent jsr:@rullama/tool-builtins
```

## Quick Example

```ts
import { Task } from "@rullama/core";
import { CommunicationHub, FileLockManager } from "@rullama/agent";
import {
  AgentContext,
  AgentPool,
  parseVerdict,
  TaskAgent,
  verdictType,
} from "@rullama/inference";
import { AgentCapabilities } from "@rullama/permission";
import { AnthropicChatProvider } from "@rullama/provider";
import { createBuiltinExecutor } from "@rullama/tool-builtins";

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
);

// AgentContext wraps the executor in @rullama/tool-runtime's enforcing
// executor by default (permission mode, capabilities, policy engine, output
// filtering). The optional sixth argument tunes it; `false` disables it.
const context = new AgentContext(
  Deno.cwd(),
  createBuiltinExecutor(),
  new CommunicationHub(),
  new FileLockManager(),
  undefined,
  { mode: "auto", capabilities: AgentCapabilities.standardDev() },
);

// One agent, one task
const task = new Task("task-1", "Add a --verbose flag to the CLI");
const agent = new TaskAgent("worker-1", task, provider, context, {
  maxIterations: 25,
  systemPrompt: "You are a careful coding agent.",
});
const result = await agent.execute();
console.log(result.success, result.summary, result.totalTokensUsed);

// Several agents in parallel
const pool = new AgentPool(3, provider, context);
const id = pool.spawnAgent(new Task("task-2", "Write tests for the parser"));
const outcome = await pool.awaitCompletion(id);
console.log(outcome.success, pool.stats());

// Judge output parsing (drive a TaskAgent with `judgeAgentPrompt` first)
const verdict = parseVerdict('{"type":"approve","summary":"looks good"}');
console.log(verdictType(verdict));
```

## Sub-path exports

`@rullama/inference/task-agent`, `/judge`, `/planner`, `/validator`, `/cycle`,
`/runtime`.
