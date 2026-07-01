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

### Core

| Example                                               | Description                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| [quickstart.ts](./core/quickstart.ts)                 | Implement the Provider interface and send a chat request                       |
| [agent_quickstart.ts](./core/agent_quickstart.ts)     | Set up agent infrastructure with tasks, lifecycle hooks, and working set       |
| [tool_usage.ts](./core/tool_usage.ts)                 | Define tools, create tool results, and use the idempotency registry            |
| [rag_pipeline.ts](./core/rag_pipeline.ts)             | Implement EmbeddingProvider and VectorStore for retrieval-augmented generation |
| [storage_and_search.ts](./core/storage_and_search.ts) | Use output parsers, plans, error handling, and content source trust            |
| [streaming.ts](./core/streaming.ts)                   | Implement and consume a streaming provider with chunk handling                 |

### Tool System

| Example                                                    | Description                                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [tool_registry.ts](./tool-system/tool_registry.ts)         | Create a ToolRegistry, register tools, list by category, and search                   |
| [tool_execution.ts](./tool-system/tool_execution.ts)       | ToolExecutor and ToolPreHook for pre-execution validation and audit logging           |
| [tool_filtering.ts](./tool-system/tool_filtering.ts)       | Sanitization, error classification, injection detection, and sensitive data redaction |
| [tool_transactions.ts](./tool-system/tool_transactions.ts) | TransactionManager for two-phase commit file write operations                         |
| [smart_routing.ts](./tool-system/smart_routing.ts)         | Smart tool router that analyzes queries to determine relevant tool categories         |

### Providers

| Example                                                | Description                                                                 |
| ------------------------------------------------------ | --------------------------------------------------------------------------- |
| [provider_factory.ts](./providers/provider_factory.ts) | Browse the provider registry, build configs, and inspect model capabilities |
| [rate_limiting.ts](./providers/rate_limiting.ts)       | RateLimiter and RateLimitedClient for token-bucket API throttling           |

### Storage

| Example                                                | Description                                                          |
| ------------------------------------------------------ | -------------------------------------------------------------------- |
| [message_store.ts](./storage/message_store.ts)         | InMemoryMessageStore for conversation messages with search           |
| [tiered_memory.ts](./storage/tiered_memory.ts)         | Hot/warm/cold memory hierarchy with importance scores and demotion   |
| [plan_templates.ts](./storage/plan_templates.ts)       | TemplateStore for reusable plan templates with variable substitution |
| [lock_coordination.ts](./storage/lock_coordination.ts) | In-memory resource locking for agent coordination                    |

### Permissions

| Example                                            | Description                                                                        |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [policy_engine.ts](./permissions/policy_engine.ts) | Declarative policy rules with deny, allow-with-audit, and require-approval actions |
| [trust_audit.ts](./permissions/trust_audit.ts)     | Trust level management with TrustManager and audit event logging                   |

### Cognition

| Example                                                        | Description                                                                               |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [rag_search.ts](./cognition/rag_search.ts)                     | RagClient interface for codebase indexing and hybrid semantic+keyword search              |
| [code_analysis.ts](./cognition/code_analysis.ts)               | Symbol extraction, reference finding, call graph construction, and repo map formatting    |
| [knowledge_graph.ts](./cognition/knowledge_graph.ts)           | Entity extraction, relationship modeling, thought creation, and the BrainClient interface |
| [prompting_techniques.ts](./cognition/prompting_techniques.ts) | List, group, and filter the 15 adaptive prompting techniques                              |

### Agents

| Example                                                   | Description                                                                            |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [planner_agent.ts](./agents/planner_agent.ts)             | Parse PlannerOutput and JudgeVerdict structured output formats                         |
| [validator_agent.ts](./agents/validator_agent.ts)         | ValidatorAgent for running quality gate checks on a working set                        |
| [agent_pool.ts](./agents/agent_pool.ts)                   | AgentPool for managing concurrent TaskAgents with CommunicationHub and FileLockManager |
| [task_decomposition.ts](./agents/task_decomposition.ts)   | Task decomposition strategies and MDAP cost estimation                                 |
| [voting_consensus.ts](./agents/voting_consensus.ts)       | MAKER voting consensus with sampled responses and red-flag validation                  |
| [contract_net.ts](./agents/contract_net.ts)               | Contract-Net bidding protocol with bid scoring and evaluation strategies               |
| [saga_compensation.ts](./agents/saga_compensation.ts)     | SagaExecutor with compensating transactions and automatic rollback                     |
| [optimistic_locking.ts](./agents/optimistic_locking.ts)   | Optimistic concurrency with conflict detection and resolution strategies               |
| [market_coordination.ts](./agents/market_coordination.ts) | Market-based resource allocation with dynamic urgency and agent budgets                |
| [three_state.ts](./agents/three_state.ts)                 | Three-state model separating application, operation, and dependency state              |
| [wait_queue.ts](./agents/wait_queue.ts)                   | WaitQueue for resource coordination with priority ordering and wait estimates          |
| [dag_workflow.ts](./agents/dag_workflow.ts)               | ExecutionGraph for DAG-based workflow tracking with telemetry                          |

### A2A (Agent-to-Agent)

| Example                                            | Description                                                                         |
| -------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [a2a_client_server.ts](./a2a/a2a_client_server.ts) | A2A client API for sending messages, listing tasks, and canceling tasks             |
| [agent_card.ts](./a2a/agent_card.ts)               | Build AgentCards with capabilities, skills, and security schemes                    |
| [a2a_streaming.ts](./a2a/a2a_streaming.ts)         | A2A streaming types: TaskStatusUpdateEvent, TaskArtifactUpdateEvent, StreamResponse |

### Agent Network

| Example                                                  | Description                                                                                  |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [network_manager.ts](./agent-network/network_manager.ts) | Agent identities, capability cards, messaging envelopes, and peer discovery                  |
| [mcp_server.ts](./agent-network/mcp_server.ts)           | McpToolRegistry, middleware pipeline (auth, logging, rate-limiting), and server construction |
| [peer_discovery.ts](./agent-network/peer_discovery.ts)   | Discovery layer, PeerTable management, and routing strategies                                |

### MCP

| Example                              | Description                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| [mcp_client.ts](./mcp/mcp_client.ts) | MCP client configuration, McpConfigManager, and the full McpClient API surface |

### Skills

| Example                                         | Description                                                                             |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| [skill_registry.ts](./skills/skill_registry.ts) | SKILL.md creation, skill discovery with SkillRegistry, query matching, and lazy-loading |
