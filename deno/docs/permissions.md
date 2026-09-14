# Permissions

The `@rullama/permission` package provides capability-based access control,
policy rules, audit logging, trust management and approval types. Since v0.12.1
these are **enforced**: `@rullama/tool-runtime`'s `EnforcingExecutor` consults
the permission mode, the capability profile and the policy engine before every
tool call, and `@rullama/inference`'s `AgentContext` applies it by default. See
[tools.md](./tools.md#enforcement) for the executor side; this page covers the
building blocks.

## Capability Profiles

`AgentCapabilities` bundles fine-grained controls over filesystem, tools,
network, git, spawning and resource quotas.

```ts
import {
  AgentCapabilities,
  defaultFilesystemCapabilities,
  defaultResourceQuotas,
  parseCapabilityProfile,
  standardGitCapabilities,
} from "@rullama/permission";

// Use a preset profile
const caps = AgentCapabilities.standardDev(); // or .readOnly() / .fullAccess()
const profile = parseCapabilityProfile("standard_dev"); // CapabilityProfile | undefined
const fromProfile = AgentCapabilities.fromProfile(profile!);

// Or build custom capabilities
const custom = new AgentCapabilities({
  filesystem: defaultFilesystemCapabilities(),
  git: standardGitCapabilities(),
  quotas: defaultResourceQuotas(),
});
```

Preset profiles: `read_only`, `standard_dev`, `full_access`. Pass the profile to
the executor: `enforce(executor, { capabilities: caps })` or
`new AgentContext(..., { capabilities: caps })`.

## PolicyEngine

The `PolicyEngine` evaluates a `PolicyRequest` against prioritized policies to
produce a `PolicyDecision` (`allow` / `deny` / `require_approval`, …).
`PolicyEngine.withDefaults()` is what the enforcing executor uses when you pass
none: it denies `.env`, secret and credential files and requires approval for
`git reset` / `git rebase`.

```ts
import {
  createPolicy,
  PolicyActions,
  PolicyEngine,
  policyRequestForGit,
} from "@rullama/permission";

const engine = PolicyEngine.withDefaults();

engine.addPolicy(createPolicy("no-force-push", {
  description: "Force push is prohibited",
  priority: 100,
  conditions: [{ type: "git_op", operation: "ForcePush" }],
  action: PolicyActions.DenyWithMessage("Force push is prohibited"),
}));

const decision = engine.evaluate(policyRequestForGit("ForcePush"));
console.log(decision.action.type, decision.reason);
```

Conditions (`PolicyCondition`): `tool`, `tool_category`, `file_path`,
`min_trust_level`, `domain`, `git_op`, `time_range`, `and`, `or`, `not`,
`always`. Actions (`PolicyActions`): `Allow`, `Deny`, `RequireApproval`,
`AllowWithAudit`, `DenyWithMessage(msg)`, `Escalate`.

Helpers for common requests: `createPolicyRequest`, `policyRequestForTool`,
`policyRequestForFile`, `policyRequestForNetwork`, `policyRequestForGit`; the
executor-side `policyRequestForToolUse` (in `@rullama/tool-runtime`) builds one
from a `ToolUse`.

See: `../examples/permissions/policy_engine.ts`.

## TrustManager

`TrustManager` tracks a `TrustFactor` per agent from success / failure history
and violation severity, and maps the score to a `TrustLevel` (`untrusted` /
`low` / `medium` / `high` / `system`).

```ts
import { TrustManager } from "@rullama/permission";

const manager = TrustManager.inMemory(); // or TrustManager.withPath(file)

manager.recordSuccess("worker-1");
manager.recordViolation("worker-1", "major"); // "minor" | "major" | "critical"

const level = manager.getTrustLevel("worker-1");
const factor = manager.get("worker-1"); // TrustFactor | undefined
const stats = manager.statistics(); // TrustStatistics
```

Types: `TrustFactor`, `TrustLevel`, `ViolationSeverity`, `ViolationCounts`,
`TrustStatistics`.

## AuditLogger

`AuditLogger` records and queries security-relevant events.

```ts
import {
  AuditLogger,
  createAuditEvent,
  createAuditQuery,
  withAction,
  withAgent,
  withOutcome,
} from "@rullama/permission";

const logger = AuditLogger.create(); // or AuditLogger.withPath(file)

logger.log(
  withOutcome(
    withAction(
      withAgent(createAuditEvent("tool_execution"), "worker-1"),
      "bash",
    ),
    "success",
  ),
);
logger.logToolExecution("worker-1", "bash", undefined, "success", 12);

const events = logger.query(createAuditQuery({ agent_id: "worker-1" }));
const stats = logger.statistics(); // AuditStatistics
```

Wire the enforcing executor's `onDecision` callback to the logger to record
every allow / deny / approval decision.

See: `../examples/permissions/trust_audit.ts`.

## Anomaly Detection

`AnomalyDetector` lives in **`@rullama/telemetry`** (not in this package). It
watches a stream of `ObservedEvent`s for unusual action frequencies and sudden
behaviour changes.

```ts
import { AnomalyDetector, defaultAnomalyConfig } from "@rullama/telemetry";

const detector = new AnomalyDetector(defaultAnomalyConfig());
detector.observe({
  timestamp: new Date().toISOString(),
  event_type: "policy_violation",
  agent_id: "worker-1",
});
const anomalies = detector.drainAnomalies(); // AnomalyEvent[]
```

## Approval Workflows

A `require_approval` policy decision (or, in `"auto"` mode, a tool flagged
`requires_approval`) is routed by the enforcing executor to its
`ApprovalHandler` --
`(toolUse, decision, context) => boolean | Promise<boolean>` -- and denied when
no handler is configured. The structured request/response types are here:

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

- [Tools](./tools.md) for the enforcing executor and built-in tool limits
- [Agents](./agents.md) for `AgentContext` enforcement defaults
- [Extensibility](./extensibility.md) for custom policy conditions
