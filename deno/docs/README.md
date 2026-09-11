# rullama Documentation

Guides for the Deno/TypeScript port of the rullama framework (27 packages,
`@rullama/*` on JSR).

## Guides

- [Getting Started](./getting-started.md) -- Install, configure, and run your
  first agent in 5 minutes
- [Architecture](./architecture.md) -- Package dependency graph, design
  philosophy, and package overview
- [Agents](./agents.md) -- Agent loop, `TaskAgent`, `AgentContext`, coordination
  patterns, validator / judge / planner helpers, and MDAP voting
  (`@rullama/inference`, `@rullama/agent`, `@rullama/mdap`)
- [Providers](./providers.md) -- Provider interface, factory, streaming, rate
  limiting, call policies (`@rullama/provider`, `@rullama/call-policy`,
  `@rullama/provider-speech`)
- [Tools](./tools.md) -- Enforcing executor, built-in tools and their safety
  limits, registry, smart routing, transactions (`@rullama/tool-runtime`,
  `@rullama/tool-builtins`)
- [Permissions](./permissions.md) -- Capability profiles, policy engine, trust,
  audit, approvals (`@rullama/permission`)
- [Storage](./storage.md) -- Storage backends, raw-filter and identifier guards,
  domain stores, tiered memory (`@rullama/storage`, `@rullama/stores`,
  `@rullama/memory`)
- [Prompting](./prompting.md) -- Adaptive prompting techniques, clustering,
  temperature, and the `BrainClient` knowledge contract (`@rullama/prompting`,
  `@rullama/knowledge`)
- [RAG](./rag.md) -- `RagClient` and code analysis (`@rullama/rag`)
- [Networking](./networking.md) -- MCP server framework, middleware, routing,
  discovery, remote bridge (`@rullama/mcp-server`, `@rullama/network`)
- [A2A Protocol](./a2a.md) -- Agent-to-Agent client, agent cards, task
  lifecycle, and streaming (`@rullama/a2a`)
- [Extensibility](./extensibility.md) -- Key interfaces to implement, extension
  patterns, and custom backends
- [Parity](./parity.md) -- Crate-by-crate diff against the Rust workspace

## Packages without a dedicated guide

Each package README documents its own surface:

- [`@rullama/core`](../packages/core/README.md) -- foundation types (messages,
  tools, errors, lifecycle hooks, working set, confidence, paths)
- [`@rullama/eval`](../packages/eval/README.md) -- evaluation harness (Monte
  Carlo suites, Wilson CI, adversarial / regression / stability cases, ranking
  metrics)
- [`@rullama/finetune`](../packages/finetune/README.md) -- cloud fine-tuning
  (OpenAI, Together, Fireworks)
- [`@rullama/mcp-client`](../packages/mcp-client/README.md) -- MCP client over
  stdio + `McpConfigManager`
- [`@rullama/reasoning`](../packages/reasoning/README.md) -- Tier-1 local
  scorers and the plan parser
- [`@rullama/seal`](../packages/seal/README.md) -- SEAL self-evolving learning
  loop
- [`@rullama/session`](../packages/session/README.md) -- session persistence
  (in-memory, Deno KV)
- [`@rullama/skills`](../packages/skills/README.md) -- SKILL.md registry,
  router, executor
- [`@rullama/telemetry`](../packages/telemetry/README.md) -- analytics events,
  sinks, Prometheus metrics, billing hooks, anomaly detection

## Quick Links

- [Main README](../README.md) -- Package listing and installation
- [Examples](../examples/) -- Runnable example scripts for every package
