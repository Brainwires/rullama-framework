# rullama — Deno Examples

Runnable examples demonstrating every major package in the rullama framework.
Each file is self-contained and uses in-memory mocks so you can run them without
external services.

## How to run

```bash
deno run deno/examples/<path>.ts
```

Some examples require additional permissions:

```bash
deno run --allow-read --allow-write --allow-env deno/examples/<path>.ts
```

Check the `// Run:` comment at the top of each file for the exact command.

---

## Table of Contents

### Core (`@rullama/core`)

| Example                                               | Description                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| [quickstart.ts](./core/quickstart.ts)                 | Implement the Provider interface and send a chat request                       |
| [agent_quickstart.ts](./core/agent_quickstart.ts)     | Set up agent infrastructure with tasks, lifecycle hooks, and working set       |
| [tool_usage.ts](./core/tool_usage.ts)                 | Define tools, create tool results, and use the idempotency registry            |
| [rag_pipeline.ts](./core/rag_pipeline.ts)             | Implement EmbeddingProvider and VectorStore for retrieval-augmented generation |
| [storage_and_search.ts](./core/storage_and_search.ts) | Use output parsers, plans, error handling, and content source trust            |
| [streaming.ts](./core/streaming.ts)                   | Implement and consume a streaming provider with chunk handling                 |

### Tools (`@rullama/tool-runtime`, `@rullama/tool-builtins`)

| Example                                              | Description                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [tool_registry.ts](./tools/tool_registry.ts)         | Create a ToolRegistry, register built-in and custom tools, list by category, search   |
| [tool_execution.ts](./tools/tool_execution.ts)       | ToolExecutor and ToolPreHook for pre-execution validation and audit logging           |
| [tool_filtering.ts](./tools/tool_filtering.ts)       | Sanitization, error classification, injection detection, and sensitive data redaction |
| [tool_transactions.ts](./tools/tool_transactions.ts) | TransactionManager for two-phase commit file write operations                         |
| [smart_routing.ts](./tools/smart_routing.ts)         | Smart tool router that analyzes queries to determine relevant tool categories         |

### Providers (`@rullama/provider`)

| Example                                                | Description                                                                               |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| [provider_factory.ts](./providers/provider_factory.ts) | Browse the provider registry, build configs, create providers, inspect model capabilities |
| [rate_limiting.ts](./providers/rate_limiting.ts)       | RateLimiter and RateLimitedClient for token-bucket API throttling                         |

### Storage (`@rullama/storage`, `@rullama/stores`, `@rullama/memory`)

| Example                                                | Description                                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| [message_store.ts](./storage/message_store.ts)         | `InMemoryMessageStore` (`@rullama/stores`) for conversation messages with search            |
| [tiered_memory.ts](./storage/tiered_memory.ts)         | `TieredMemory` (`@rullama/memory`) hot/warm/cold hierarchy with importance scores, demotion |
| [plan_templates.ts](./storage/plan_templates.ts)       | `TemplateStore` (`@rullama/stores`) for reusable plan templates with variable substitution  |
| [lock_coordination.ts](./storage/lock_coordination.ts) | Resource locking across agents on top of `InMemoryStorageBackend` (`@rullama/storage`)      |

### Permissions (`@rullama/permission`)

| Example                                            | Description                                                                        |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [policy_engine.ts](./permissions/policy_engine.ts) | Declarative policy rules with deny, allow-with-audit, and require-approval actions |
| [trust_audit.ts](./permissions/trust_audit.ts)     | Trust level management with TrustManager and audit event logging                   |

### Knowledge (`@rullama/knowledge`, `@rullama/rag`, `@rullama/prompting`)

| Example                                                        | Description                                                                                           |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [knowledge_graph.ts](./knowledge/knowledge_graph.ts)           | Entity extraction, relationship modeling, thought creation, and the BrainClient interface (knowledge) |
| [rag_search.ts](./knowledge/rag_search.ts)                     | RagClient interface for codebase indexing and hybrid semantic+keyword search (rag)                    |
| [code_analysis.ts](./knowledge/code_analysis.ts)               | Symbol extraction, reference finding, call graph construction, and repo map formatting (rag)          |
| [prompting_techniques.ts](./knowledge/prompting_techniques.ts) | List, group, and filter the 15 adaptive prompting techniques (prompting)                              |

### Agents (`@rullama/agent`, `@rullama/inference`, `@rullama/mdap`, `@rullama/skills`)

| Example                                                   | Description                                                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [planner_agent.ts](./agents/planner_agent.ts)             | Parse PlannerOutput and JudgeVerdict structured output formats (inference)                       |
| [validator_agent.ts](./agents/validator_agent.ts)         | ValidatorAgent for running quality gate checks on a working set (inference)                      |
| [agent_pool.ts](./agents/agent_pool.ts)                   | AgentPoolStats (inference) plus CommunicationHub and FileLockManager (agent)                     |
| [task_decomposition.ts](./agents/task_decomposition.ts)   | Task decomposition strategies and MDAP cost estimation (mdap)                                    |
| [voting_consensus.ts](./agents/voting_consensus.ts)       | MAKER voting consensus with sampled responses and red-flag validation (mdap)                     |
| [skill_registry.ts](./agents/skill_registry.ts)           | SKILL.md creation, skill discovery with SkillRegistry, query matching, and lazy-loading (skills) |
| [contract_net.ts](./agents/contract_net.ts)               | Contract-Net bidding protocol with bid scoring and evaluation strategies                         |
| [saga_compensation.ts](./agents/saga_compensation.ts)     | SagaExecutor with compensating transactions and automatic rollback                               |
| [optimistic_locking.ts](./agents/optimistic_locking.ts)   | Optimistic concurrency with conflict detection and resolution strategies                         |
| [market_coordination.ts](./agents/market_coordination.ts) | Market-based resource allocation with dynamic urgency and agent budgets                          |
| [three_state.ts](./agents/three_state.ts)                 | Three-state model separating application, operation, and dependency state                        |
| [wait_queue.ts](./agents/wait_queue.ts)                   | WaitQueue for resource coordination with priority ordering and wait estimates                    |
| [dag_workflow.ts](./agents/dag_workflow.ts)               | ExecutionGraph for DAG-based workflow tracking with telemetry                                    |

### A2A (`@rullama/a2a`)

| Example                                            | Description                                                                         |
| -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [a2a_client_server.ts](./a2a/a2a_client_server.ts) | A2A client API for sending messages, listing tasks, and canceling tasks             |
| [agent_card.ts](./a2a/agent_card.ts)               | Build AgentCards with capabilities, skills, and security schemes                    |
| [a2a_streaming.ts](./a2a/a2a_streaming.ts)         | A2A streaming types: TaskStatusUpdateEvent, TaskArtifactUpdateEvent, StreamResponse |

### Network (`@rullama/network`, `@rullama/mcp-server`)

| Example                                            | Description                                                                                               |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [network_manager.ts](./network/network_manager.ts) | Agent identities, capability cards, messaging envelopes, and peer discovery (network)                     |
| [peer_discovery.ts](./network/peer_discovery.ts)   | Discovery layer, PeerTable management, and routing strategies (network)                                   |
| [mcp_server.ts](./network/mcp_server.ts)           | McpToolRegistry, middleware pipeline (auth, logging, rate-limiting), and server construction (mcp-server) |

### MCP client (`@rullama/mcp-client`)

| Example                              | Description                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| [mcp_client.ts](./mcp/mcp_client.ts) | MCP client configuration, McpConfigManager, and the full McpClient API surface |
