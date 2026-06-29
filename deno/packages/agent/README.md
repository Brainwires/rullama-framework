# @rullama/agents

Agent orchestration, coordination, and lifecycle management for the Brainwires
Agent Framework. Provides the core agentic execution loop, task agents,
inter-agent communication, and distributed coordination patterns.

Equivalent to the Rust `rullama-agent` crate.

## Install

```sh
deno add @rullama/agents
```

## Quick Example

```ts
import { ChatOptions, Message } from "@rullama/core";
import { AnthropicChatProvider } from "@rullama/provider";
import { BashTool, ToolRegistry } from "@rullama/tools";
import { AgentContext, spawnTaskAgent, TaskAgent } from "@rullama/agent";

const registry = new ToolRegistry();
registry.registerTools(BashTool.getTools());

const provider = new AnthropicChatProvider(
  Deno.env.get("ANTHROPIC_API_KEY")!,
  "claude-sonnet-4-20250514",
  "anthropic",
);

const context = new AgentContext({ tools: registry.allTools() });

const result = await spawnTaskAgent({
  agentId: "my-agent",
  provider,
  context,
  systemPrompt: "You are a helpful assistant.",
  taskDescription: "Show the current date and time.",
});

console.log(result.output);
```

## Core Components

| Component                      | Description                                                               |
| ------------------------------ | ------------------------------------------------------------------------- |
| `runAgentLoop`                 | Generic execution loop: call provider, extract tool uses, execute, repeat |
| `TaskAgent` / `spawnTaskAgent` | Concrete agent with provider + tool loop and validation                   |
| `AgentContext`                 | Environment bundle (tools, communication hub, file locks, working set)    |
| `CommunicationHub`             | Inter-agent messaging bus with conflict detection                         |
| `FileLockManager`              | File access coordination with deadlock detection                          |
| `TaskManager`                  | Hierarchical task decomposition and dependency tracking                   |
| `TaskQueue`                    | Priority-based task scheduling                                            |
| `PlanExecutorAgent`            | Plan step execution orchestration                                         |

## Coordination Patterns

| Pattern                | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `ContractNetManager`   | Bidding protocol for task-to-agent negotiation            |
| `SagaExecutor`         | Compensating transactions for multi-step operations       |
| `OptimisticController` | Optimistic locking with conflict detection and resolution |
