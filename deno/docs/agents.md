# Agents

The agent system spans three packages: `@rullama/inference` (agent runtime,
`TaskAgent`, pool, validators and the judge/planner helpers), `@rullama/agent`
(multi-agent coordination primitives and patterns), and `@rullama/mdap` (the
MDAP/MAKER voting framework).

## Agent Loop

The core execution model is `runAgentLoop`, which drives any `AgentRuntime`
implementation through an iterate-until-done loop with tool calling,
communication-hub registration and file locking.

```ts
import { type AgentRuntime, runAgentLoop } from "@rullama/inference";
import { CommunicationHub, FileLockManager } from "@rullama/agent";

const hub = new CommunicationHub();
const lockManager = new FileLockManager();
const result = await runAgentLoop(myRuntime, hub, lockManager);
```

`AgentRuntime` requires: `agentId`, `maxIterations`, `callProvider` (returns a
`ChatResponse`), `extractToolUses`, `isCompletion`, `executeTool`,
`getLockRequirement`, `onProviderResponse`, `onToolResult`, `onCompletion`, and
`onIterationLimit`; `lifecycleHooks`, `contextBudgetTokens` and `conversation`
are optional.

## AgentContext

`AgentContext` bundles everything an agent needs: the working directory, a
`ToolExecutor`, the `CommunicationHub`, the `FileLockManager`, a `WorkingSet`,
metadata, and optional pre-execute / lifecycle hooks. Arguments are positional.

**Since v0.12.1 the context enforces permissions by default:** the executor you
pass is wrapped in `@rullama/tool-runtime`'s `EnforcingExecutor` (permission
mode, capability profile, `PolicyEngine.withDefaults()`, pre-execute hooks,
output filtering). The optional sixth argument tunes that wrapper, or
`enforcement: false` disables it.

```ts
import { CommunicationHub, FileLockManager } from "@rullama/agent";
import { AgentContext } from "@rullama/inference";
import { AgentCapabilities } from "@rullama/permission";
import { createBuiltinExecutor } from "@rullama/tool-builtins";

const context = new AgentContext(
  Deno.cwd(),
  createBuiltinExecutor(), // bash, file ops, git, web, search
  new CommunicationHub(),
  new FileLockManager(),
  undefined, // WorkingSet (defaults to a new one)
  { mode: "auto", capabilities: AgentCapabilities.standardDev() },
);
context.withPreExecuteHook(myHook).withLifecycleHooks(myLifecycleHooks);
```

See [tools.md](./tools.md#enforcement) for the enforcement model.

## TaskAgent and spawnTaskAgent

`TaskAgent` is the concrete agent that wires a `Provider`, an `AgentContext` and
a `Task` into a working runtime. `spawnTaskAgent` creates one and starts
`execute()` in a single call.

```ts
import { Task } from "@rullama/core";
import { spawnTaskAgent, TaskAgent } from "@rullama/inference";
import { AnthropicChatProvider } from "@rullama/provider";

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
);
const task = new Task("task-1", "Refactor the utils module.");

const agent = new TaskAgent("worker-1", task, provider, context, {
  maxIterations: 20,
});
const result = await agent.execute();

// or: spawn + run
const { agent: worker, result: pending } = spawnTaskAgent(
  "worker-2",
  task,
  provider,
  context,
);
console.log(worker.status, (await pending).success);
```

Key types: `TaskAgentConfig`, `TaskAgentResult`, `TaskAgentStatus`,
`LoopDetectionConfig`, `FailureCategory`.

## AgentPool

`AgentPool` runs up to N `TaskAgent`s concurrently over a shared provider and
context: `spawnAgent`, `getStatus`, `awaitCompletion`, `awaitAll`,
`cleanupCompleted`, `shutdown`, and `stats()` (an `AgentPoolStats`).

See: `../examples/agents/agent_pool.ts`.

## Coordination Patterns (`@rullama/agent`)

| Pattern                | Class                  | Purpose                                                           |
| ---------------------- | ---------------------- | ----------------------------------------------------------------- |
| Contract Net           | `ContractNetManager`   | Bidding protocol -- announce tasks, collect bids, award contracts |
| Saga                   | `SagaExecutor`         | Compensating transactions with rollback on failure                |
| Optimistic Concurrency | `OptimisticController` | Version-based locking with conflict detection                     |
| Market Allocator       | `MarketAllocator`      | Budget-based task allocation with pricing strategies              |
| Three-State Model      | `ThreeStateModel`      | State snapshots, operation logs, and rollback                     |
| Wait Queue             | `WaitQueue`            | Queue-based resource synchronization                              |

Plus `CommunicationHub`, `FileLockManager`, `TaskManager` / `TaskQueue` and
`ExecutionGraph` (DAG telemetry).

See examples: `../examples/agents/contract_net.ts`,
`../examples/agents/saga_compensation.ts`,
`../examples/agents/optimistic_locking.ts`,
`../examples/agents/market_coordination.ts`,
`../examples/agents/three_state.ts`, `../examples/agents/wait_queue.ts`,
`../examples/agents/dag_workflow.ts`.

## Validation, Judge and Planner helpers

There are **no** `JudgeAgent`, `PlannerAgent` or `CycleOrchestrator` classes in
the Deno port (the Rust crate's `chat_agent` / `cycle_orchestrator` are not
ported). What ships:

- **`ValidatorAgent`** -- a class that runs read-only quality gates
  (`runValidation` with `ValidationCheck`s such as `no_duplicates`,
  `syntax_valid`, `build_success`) with a timeout, plus
  `formatValidationFeedback` / `formatValidatorStatus`.
- **`PlanExecutorAgent`** -- executes a plan step by step under an
  `ExecutionApprovalMode` (`suggest` / `auto_edit` / `full_auto`).
- **Judge helpers** -- `judgeAgentPrompt`, `buildJudgeTaskDescription`,
  `parseVerdict` (→ `JudgeVerdict`), `verdictType`, `verdictHints`,
  `JudgeAgentConfig` / `defaultJudgeAgentConfig`. Run a `TaskAgent` with the
  judge prompt and parse its output.
- **Planner helpers** -- `plannerAgentPrompt`, `parsePlannerOutput` (→
  `PlannerOutput` with `DynamicTaskSpec`s), `validateTaskGraph` (cycle
  detection), `PlannerAgentConfig` / `defaultPlannerAgentConfig`.
- **Cycle orchestration** -- `CycleOrchestratorConfig`, `CycleRecord`,
  `CycleOrchestratorResult`, `MergeStrategy`, `FailurePolicy` are configuration
  and result **types only**; the Plan → Work → Judge loop is yours to drive.
- **`AgentRole`** -- `allowedTools` / `filterTools` / `systemPromptSuffix` for
  least-privilege tool sets per role.

See: `../examples/agents/planner_agent.ts`,
`../examples/agents/validator_agent.ts`.

## MDAP (MAKER Voting Framework)

MDAP samples multiple LLM responses and uses first-to-ahead-by-k voting to
select the best one. Key components:

- `FirstToAheadByKVoter` / `VoterBuilder` -- consensus voting with early
  stopping, Borda count and confidence weighting
- `StandardRedFlagValidator` (and `AcceptAllValidator`) -- output validation
- `Composer` -- combines subtask outputs into a final result
- `MdapMetrics` -- execution metrics and cost tracking
- Scaling laws: `calculateKMin`, `estimateMdap`, `calculateExpectedCost`,
  `suggestKForBudget`
- Decomposition: `atomicDecomposition`, `compositeDecomposition`,
  `createAtomicSubtask`, `topologicalSort`

```ts
import { defaultEarlyStopping, VoterBuilder } from "@rullama/mdap";

const voter = new VoterBuilder()
  .k(2)
  .maxSamples(7)
  .earlyStopping(defaultEarlyStopping())
  .confidenceWeighted(true)
  .build();
```

See: `../examples/agents/voting_consensus.ts`,
`../examples/agents/task_decomposition.ts`.

## Further Reading

- [Providers](./providers.md) for configuring the AI backend
- [Tools](./tools.md) for the tool executor and enforcement
- [Extensibility](./extensibility.md) for custom agent runtimes
