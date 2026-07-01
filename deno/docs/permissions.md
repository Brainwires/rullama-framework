# Permissions

The `@rullama/permissions` package provides capability-based access control,
policy enforcement, audit logging, trust management, and anomaly detection.

## Capability Profiles

`AgentCapabilities` bundles fine-grained controls over filesystem, tools,
network, git, spawning, and resource quotas.

```ts
import {
  AgentCapabilities,
  defaultFilesystemCapabilities,
  defaultResourceQuotas,
  parseCapabilityProfile,
  standardGitCapabilities,
} from "@rullama/permission";

// Use a preset profile
const caps = parseCapabilityProfile("standard_dev");

// Or build custom capabilities
const custom = new AgentCapabilities({
  filesystem: defaultFilesystemCapabilities(),
  git: standardGitCapabilities(),
  quotas: defaultResourceQuotas(),
});
```

Preset profiles: `read_only`, `standard_dev`, `full_access`.

## PolicyEngine

The `PolicyEngine` evaluates requests against a set of rules to produce
allow/deny/requires-approval decisions.

```ts
import {
  createPolicy,
  createPolicyRequest,
  PolicyActions,
  PolicyEngine,
} from "@rullama/permission";

const engine = new PolicyEngine();

// Add a policy
engine.addPolicy(createPolicy({
  name: "no-force-push",
  conditions: [{
    field: "action",
    operator: "equals",
    value: "git_force_push",
  }],
  action: PolicyActions.deny("Force push is prohibited"),
}));

// Evaluate a request
const request = createPolicyRequest({
  action: "git_force_push",
  agent: "worker-1",
});
const decision = engine.evaluate(request);
```

Helpers for common request types: `policyRequestForFile`, `policyRequestForGit`,
`policyRequestForNetwork`, `policyRequestForTool`.

See: `../examples/permissions/policy_engine.ts`.

## TrustManager

`TrustManager` tracks agent trust levels based on success/failure history and
violation severity.

```ts
import {
  createTrustFactor,
  trustLevelFromScore,
  TrustManager,
} from "@rullama/permission";

const manager = new TrustManager();
manager.registerAgent("worker-1", createTrustFactor("worker-1"));

// Record outcomes
manager.recordSuccess("worker-1");
manager.recordViolation("worker-1", "medium");

// Check trust
const factor = manager.getTrustFactor("worker-1");
const level = trustLevelFromScore(factor.score); // "high", "medium", "low", "untrusted"
```

Types: `TrustFactor`, `TrustLevel`, `ViolationSeverity`, `ViolationCounts`.

## AuditLogger

`AuditLogger` records and queries security-relevant events.

```ts
import {
  AuditLogger,
  createAuditEvent,
  createAuditQuery,
  withAgent,
} from "@rullama/permission";

const logger = new AuditLogger();

// Log an event
logger.log(createAuditEvent({
  type: "tool_execution",
  agent: "worker-1",
  action: "bash",
  outcome: "success",
}));

// Query events
const query = withAgent(createAuditQuery(), "worker-1");
const events = logger.query(query);
const stats = logger.statistics();
```

See: `../examples/permissions/trust_audit.ts`.

## Anomaly Detection

`AnomalyDetector` monitors the audit stream for statistical anomalies -- unusual
action frequencies, time-of-day patterns, and sudden behavior changes.

```ts
import { AnomalyDetector, defaultAnomalyConfig } from "@rullama/permission";

const detector = new AnomalyDetector(defaultAnomalyConfig());
// Feed audit events into the detector
// detector.observe(event);
// const anomalies = detector.detect();
```

Types: `AnomalyConfig`, `AnomalyEvent`, `AnomalyKind`.

## Approval Workflows

For sensitive operations, policies can return a "requires approval" decision.
The approval system provides structured request/response types:

```ts
import type { ApprovalRequest, ApprovalResponse } from "@rullama/permission";
import {
  approvalActionSeverity,
  isApprovalResponseApproved,
} from "@rullama/permission";
```

## Configuration

Load permissions from a JSON config file:

```ts
import {
  configToCapabilities,
  loadPermissionsConfig,
} from "@rullama/permission";

const config = loadPermissionsConfig("./permissions.json");
const caps = configToCapabilities(config);
```

## Further Reading

- [Agents](./agents.md) for integrating permissions into agent loops
- [Extensibility](./extensibility.md) for custom policy conditions
