# @rullama/agent

Multi-agent coordination primitives for the rullama framework: an in-memory
inter-agent message bus, read/write file locks with deadlock detection,
hierarchical task tracking with a priority queue, a per-run execution graph, and
six coordination patterns (Contract-Net, Saga, optimistic concurrency, market
allocation, the SagaLLM three-state model, and a priority wait queue).

Equivalent to the Rust `rullama-agent` crate. Everything here is single-process
and in-memory; nothing is persisted.

The LLM-driven agents themselves (`runAgentLoop`, `TaskAgent`, `spawnTaskAgent`,
`AgentContext`, `PlanExecutorAgent`, `AgentPool`) live in `@rullama/inference`,
not here.

## Install

```sh
deno add jsr:@rullama/agent
```

## Quick Example

Decompose work into dependent tasks, schedule the ready ones through a priority
queue, and coordinate file access between two agents:

```ts
import {
  CommunicationHub,
  FileLockManager,
  TaskManager,
  TaskQueue,
} from "@rullama/agent";

// Hierarchical tasks with dependencies.
const tasks = new TaskManager();
const root = tasks.createTask("Ship the feature");
const write = tasks.createTask("Write the code", root);
const test = tasks.createTask("Run the tests", root);
tasks.addDependency(test, write); // tests wait for the code

// Only dependency-free tasks are ready; queue them by priority.
const queue = new TaskQueue(50);
for (const task of tasks.getReadyTasks()) queue.enqueue(task, "high");

const next = queue.dequeueAndAssign("agent-1");
if (next) tasks.startTask(next.task.id);

// Coordinate file access: write locks exclude other agents.
const locks = new FileLockManager({ timeoutMs: 60_000 });
const guard = locks.acquireLock("agent-1", "src/feature.ts", "write");
console.log(locks.canAcquire("src/feature.ts", "agent-2", "read")); // false
guard.release();

// Tell the other agent what happened.
const hub = new CommunicationHub();
hub.registerAgent("agent-1");
hub.registerAgent("agent-2");
hub.sendMessage("agent-1", "agent-2", {
  type: "status_update",
  agentId: "agent-1",
  status: "done",
});
console.log(hub.tryReceiveMessage("agent-2")?.message);

tasks.completeTask(write, "implemented");
console.log(tasks.getStats()); // { total: 3, pending: 1, inProgress: 1, completed: 1, ... }
```

## Core Components

| Component          | Description                                                                        |
| ------------------ | ---------------------------------------------------------------------------------- |
| `CommunicationHub` | Per-agent in-memory message channels; `sendMessage`, `broadcast`, `receiveMessage` |
| `AgentMessage`     | Discriminated union of every message kind (status, help, saga, bids, conflicts)    |
| `FileLockManager`  | Read/write file locks with timeouts, polling `acquireWithWait`, deadlock detection |
| `TaskManager`      | Task tree with parent/child links, dependencies, `getReadyTasks`, stats and timing |
| `TaskQueue`        | Bounded priority queue (`urgent > high > normal > low`, FIFO within a level)       |
| `ExecutionGraph`   | Per-run step/tool-call trace; `telemetryFromGraph` summarizes it as `RunTelemetry` |

`Task`, `TaskPriority` and `TaskStatus` are re-exported (as types) from
`@rullama/core` for convenience; construct `Task` via `TaskManager.createTask`
or import the class from `@rullama/core`.

## Coordination Patterns

| Pattern                                      | Description                                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------- |
| `ContractNetManager` / `ContractParticipant` | Announce → bid → award protocol with pluggable `BidEvaluationStrategy`                |
| `SagaExecutor` / `CompensationReport`        | Run `CompensableOperation` steps; `compensateAll` undoes them in reverse              |
| `OptimisticController`                       | Version tokens, commit-time conflict detection, `ResolutionStrategy` per resource     |
| `MarketAllocator`                            | Budgeted bidding with `PricingStrategy` (first/second/fixed/dynamic/free) and urgency |
| `ThreeStateModel`                            | Application / operation / dependency state with validation and deadlock checks        |
| `WaitQueue`                                  | Priority-ordered waiters per resource key, promise-based readiness, wait estimates    |
