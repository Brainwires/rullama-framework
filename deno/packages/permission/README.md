# @rullama/permission

Capability-based permission system for the rullama. Describes what an agent may
do across filesystem, tools, network, git and spawning — with rule-based
policies, audit logging, trust tracking and approval workflows.

Equivalent to the Rust `rullama-permissions` crate. (Anomaly detection moved to
`@rullama/telemetry` in v0.11.0 and is no longer part of this package.)

## Install

```sh
deno add jsr:@rullama/permission
```

## Where enforcement happens

This package only _decides_. The decisions are applied by
`@rullama/tool-runtime`'s `EnforcingExecutor`, which wraps any `ToolExecutor`
and consults the `PolicyEngine` (and an optional `AgentCapabilities` profile)
before every tool call — `@rullama/inference` wraps its executor that way by
default. Pass your engine and capabilities through `EnforcementOptions`
(`policy`, `capabilities`); omit `policy` and `PolicyEngine.withDefaults()` is
used, which denies `.env`/secret/credential files and asks before `git reset` /
`git rebase`.

## Quick Example

```ts
import {
  AgentCapabilities,
  createPolicy,
  isDecisionAllowed,
  isDecisionRequiresApproval,
  PolicyActions,
  PolicyEngine,
  policyRequestForTool,
} from "@rullama/permission";

// Use a preset capability profile.
const capabilities = AgentCapabilities.fromProfile("standard_dev");

// A policy engine with the built-in defaults plus one custom rule.
const engine = PolicyEngine.withDefaults();
engine.addPolicy(
  createPolicy("approve-bash", {
    name: "Ask before running shell commands",
    priority: 100,
    conditions: [{ type: "tool", name: "bash" }],
    action: PolicyActions.RequireApproval,
  }),
);

// Evaluate a request. The request carries the tool name and category;
// `evaluate` returns the first matching policy's action (highest priority
// first) or the engine's default action.
const decision = engine.evaluate(policyRequestForTool("bash"));
console.log(decision.action); // { type: "require_approval" }
console.log(isDecisionAllowed(decision)); // false
console.log(isDecisionRequiresApproval(decision)); // true

// Capabilities are checked separately (the EnforcingExecutor does both).
console.log(capabilities.tools.denied_tools.has("bash")); // false
```

Wire both into tool execution:

```ts
import { enforce } from "@rullama/tool-runtime";

const executor = enforce(innerExecutor, { policy: engine, capabilities });
```

## Key Exports

| Export                                          | Description                                                                               |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `AgentCapabilities`                             | Master capability set (filesystem, tools, network, git, spawning, quotas)                 |
| `AgentCapabilities.fromProfile`                 | Build from a preset: `"read_only"`, `"standard_dev"`, `"full_access"`                     |
| `parseCapabilityProfile`                        | Parse a profile name string into a `CapabilityProfile` (or `undefined`)                   |
| `PolicyEngine`                                  | Ordered rule evaluation; `withDefaults()` ships the secret-file and destructive-git rules |
| `createPolicy(id, overrides)`                   | Build a `Policy` — supply `conditions` (AND-ed) and an `action`                           |
| `policyRequestFor{Tool,File,Network,Git}`       | Build a `PolicyRequest` for one kind of operation                                         |
| `PolicyActions`                                 | Action values: `Allow`, `Deny`, `RequireApproval`, `AllowWithAudit`, `DenyWithMessage`    |
| `AuditLogger`, `createAuditEvent`, `with*`      | Event logging with querying and statistics                                                |
| `TrustManager`                                  | Per-agent trust scores, violation tracking, optional JSON persistence                     |
| `PathPattern`                                   | Glob-based path matching for filesystem rules                                             |
| `loadPermissionsConfig`, `configToCapabilities` | Load a JSON config file and turn it into `AgentCapabilities`                              |
| `ApprovalRequest`, `ApprovalResponse`           | Types for interactive approval of sensitive operations                                    |
