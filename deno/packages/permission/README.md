# @rullama/permissions

Capability-based permission system for the Brainwires Agent Framework. Controls
what agents can do across filesystem, tools, network, git, and spawning — with
policy-based enforcement, audit logging, trust tracking, and anomaly detection.

Equivalent to the Rust `rullama-permissions` crate.

## Install

```sh
deno add @rullama/permissions
```

## Quick Example

```ts
import {
  AgentCapabilities,
  createPolicy,
  createPolicyRequest,
  parseCapabilityProfile,
  PolicyActions,
  PolicyEngine,
  policyRequestForTool,
} from "@rullama/permission";

// Use a preset capability profile
const profile = parseCapabilityProfile("standard_dev");
const capabilities = AgentCapabilities.fromProfile(profile!);

// Create a policy engine with custom rules
const engine = new PolicyEngine();
engine.addPolicy(createPolicy({
  name: "block-dangerous-bash",
  condition: { type: "tool", name: "bash" },
  action: PolicyActions.RequireApproval,
  priority: 100,
}));

// Evaluate a request
const request = policyRequestForTool("bash", "Bash");
const decision = engine.evaluate(request, capabilities);
console.log(decision); // { type: "require_approval" }
```

## Key Exports

| Export                   | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `AgentCapabilities`      | Master capability set (filesystem, tools, network, git, spawning, quotas) |
| `parseCapabilityProfile` | Load a preset: `"read_only"`, `"standard_dev"`, `"full_access"`           |
| `PolicyEngine`           | Rule-based policy evaluation with conditions and actions                  |
| `PolicyActions`          | Convenience constructors: `Allow`, `Deny`, `RequireApproval`, etc.        |
| `AuditLogger`            | Event logging with querying and statistics                                |
| `TrustManager`           | Trust levels, violation tracking, and factor management                   |
| `AnomalyDetector`        | Statistical anomaly detection on the audit stream                         |
| `PathPattern`            | Glob-based path matching for filesystem rules                             |
| Config loaders           | `loadPermissionsConfig`, `configToCapabilities`                           |
