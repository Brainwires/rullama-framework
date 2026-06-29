# Agents

The `@rullama/agents` package provides the agent runtime, task agents,
coordination patterns, specialized agents, and the MDAP voting framework.

## Agent Loop

The core execution model is `runAgentLoop`, which drives any `AgentRuntime`
implementation through an iterate-until-done loop with tool calling,
communication, and file locking.

```ts
import { type AgentRuntime, runAgentLoop } from "@rullama/agent";

const result = await runAgentLoop(myRuntime, hub, lockManager);
```

The `AgentRuntime` interface requires methods for: `agentId`, `maxIterations`,
`callProvider`, `extractToolUses`, `isCompletion`, `executeTool`,
`onProviderResponse`, `onToolResult`, `onCompletion`, and `onIterationLimit`.

## TaskAgent and spawnTaskAgent

`TaskAgent` is the concrete agent that wires a `Provider`, `AgentContext`, and
system prompt into a working runtime. The `spawnTaskAgent` helper creates and
runs one in a single call.

```ts
import { AgentContext, spawnTaskAgent } from "@rullama/agent";
import { AnthropicChatProvider } from "@rullama/provider";

const provider = new AnthropicChatProvider(
  apiKey,
  "claude-sonnet-4-20250514",
  "anthropic",
);
const context = new AgentContext({ tools: registry.allTools() });

const result = await spawnTaskAgent({
  agentId: "worker-1",
  provider,
  context,
  systemPrompt: "You are a coding agent.",
  taskDescription: "Refactor the utils module.",
});
```

Key types: `TaskAgentConfig`, `TaskAgentResult`, `TaskAgentStatus`,
`LoopDetectionConfig`.

## AgentContext

`AgentContext` bundles everything an agent needs: tools, communication hub, lock
manager, working set, and lifecycle hooks.

```ts
const context = new AgentContext({
  tools: registry.allTools(),
  hub: new CommunicationHub(),
  lockManager: new FileLockManager(),
  hooks: myLifecycleHooks,
});
```

## Coordination Patterns

Six built-in patterns for multi-agent workflows:

| Pattern                | Class                  | Purpose                                                           |
| ---------------------- | ---------------------- | ----------------------------------------------------------------- |
| Contract Net           | `ContractNetManager`   | Bidding protocol -- announce tasks, collect bids, award contracts |
| Saga                   | `SagaExecutor`         | Compensating transactions with rollback on failure                |
| Optimistic Concurrency | `OptimisticController` | Version-based locking with conflict detection                     |
| Market Allocator       | `MarketAllocator`      | Budget-based task allocation with pricing strategies              |
| Three-State Model      | `ThreeStateModel`      | State snapshots, operation logs, and rollback                     |
| Wait Queue             | `WaitQueue`            | Queue-based resource synchronization                              |

See examples: `../examples/agents/contract_net.ts`,
`../examples/agents/saga_compensation.ts`,
`../examples/agents/optimistic_locking.ts`,
`../examples/agents/market_coordination.ts`.

## Specialized Agents

- **JudgeAgent** -- LLM-powered evaluator that scores cycle outputs with
  verdicts
- **PlannerAgent** -- Dynamically generates task graphs from high-level goals
- **ValidatorAgent** -- Read-only validation with configurable checks
- **CycleOrchestrator** -- Plan, Work, Judge loop with configurable failure
  policies

See: `../examples/agents/planner_agent.ts`,
`../examples/agents/validator_agent.ts`.

## MDAP (MAKER Voting Framework)

MDAP samples multiple LLM responses and uses voting to select the best one. Key
components:

- `FirstToAheadByKVoter` -- Consensus voting with early stopping and confidence
  weighting
- `StandardRedFlagValidator` -- Output quality validation
- `Composer` -- Combines subtask outputs into a final result
- `MdapMetrics` -- Execution metrics and cost tracking
- Scaling law functions: `calculateKMin`, `estimateMdap`,
  `calculateExpectedCost`

```ts
import {
  defaultEarlyStopping,
  FirstToAheadByKVoter,
  StandardRedFlagValidator,
  VoterBuilder,
} from "@rullama/agent";

const voter = new VoterBuilder()
  .kAdvantage(2)
  .maxRounds(7)
  .earlyStopping(defaultEarlyStopping())
  .redFlagValidator(new StandardRedFlagValidator())
  .build();
```

See: `../examples/agents/voting_consensus.ts`.

## Further Reading

- [Providers](./providers.md) for configuring the AI backend
- [Tools](./tools.md) for the tool registry
- [Extensibility](./extensibility.md) for custom agent runtimes
