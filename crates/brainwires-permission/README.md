# brainwires-permission

[![Crates.io](https://img.shields.io/crates/v/brainwires-permission.svg)](https://crates.io/crates/brainwires-permission)
[![Documentation](https://img.shields.io/docsrs/brainwires-permission)](https://docs.rs/brainwires-permission)
[![License](https://img.shields.io/crates/l/brainwires-permission.svg)](LICENSE)

Capability-based permission system with policy engine, audit logging, trust management, and anomaly detection for the Brainwires Agent Framework.

## Overview

`brainwires-permission` provides a comprehensive security layer for autonomous AI agents. Every agent receives a set of capabilities that govern file access, tool usage, network calls, git operations, and child-agent spawning. A rule-based policy engine evaluates requests against configurable policies, an audit logger records every action for compliance, a trust manager tracks agent reputation over time, and an anomaly detector flags suspicious behavior in real time.

**Design principles:**

- **Capability-based** — agents receive explicit capability sets; anything not granted is denied by default
- **Defense in depth** — capabilities, policies, trust scores, and anomaly detection form independent layers
- **Observable** — every permission decision is audit-logged in JSONL for forensic review
- **WASM-compatible** — the `wasm` feature disables filesystem-dependent code (glob, dirs, persistence)

```text
  ┌──────────────────────────────────────────────────────────────┐
  │                   brainwires-permission                      │
  │                                                              │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │                  Configuration                       │    │
  │  │          (TOML · profiles · parsing)                  │    │
  │  └───────────────────────┬──────────────────────────────┘    │
  │                          │                                   │
  │  ┌───────────────┐      ▼       ┌────────────────────┐      │
  │  │  Capabilities │◄──────────►│   Policy Engine     │      │
  │  │  (6 sub-types)│             │  (rules · actions)  │      │
  │  └───────┬───────┘             └─────────┬──────────┘      │
  │          │                               │                  │
  │          │          ┌────────────────────┐│                  │
  │          │          │  Approval System   │◄                  │
  │          │          │  (async channel)   │                   │
  │          │          └────────┬───────────┘                   │
  │          │                   │                               │
  │          │          ┌────────▼───────────┐                   │
  │          │          │   Audit Logger     │                   │
  │          │          │   (JSONL · query)  │                   │
  │          │          └────────┬───────────┘                   │
  │          │                   │                               │
  │          │          ┌────────▼───────────┐                   │
  │          │          │ Anomaly Detector   │                   │
  │          │          │ (sliding windows)  │                   │
  │          │          └────────┬───────────┘                   │
  │          │                   │                               │
  │          └──────────►┌──────▼──────────┐                    │
  │                      │  Trust Manager  │                    │
  │                      │  (reputation)   │                    │
  │                      └─────────────────┘                    │
  └──────────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-permission = "0.11"
```

Create a capability set and evaluate a policy:

```rust
use brainwires_permissions::{
    AgentCapabilities, CapabilityProfile, PolicyEngine, PolicyRequest, ToolCategory,
};

// Use a built-in profile
let caps = CapabilityProfile::StandardDev.build();

// Check tool access
assert!(caps.allows_tool("read_file"));
assert!(!caps.allows_tool("bash"));

// Check file access
assert!(caps.allows_read("src/main.rs"));
assert!(!caps.allows_write(".env"));

// Evaluate a policy
let engine = PolicyEngine::with_defaults();
let request = PolicyRequest {
    tool_name: Some("write_file".into()),
    tool_category: Some(ToolCategory::FileWrite),
    file_path: Some("src/lib.rs".into()),
    ..Default::default()
};
let decision = engine.evaluate(&request);
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `native` | Yes | Native target support — enables `glob` pattern matching, `dirs` for filesystem paths, file-based persistence |
| `wasm` | No | WASM target support — disables filesystem-dependent code, uses simple string matching for paths |

```toml
# Default (native)
brainwires-permission = "0.11"

# WASM target
brainwires-permission = { version = "0.11", default-features = false, features = ["wasm"] }
```

## Architecture

### Agent Capabilities

The central type governing what an agent can do. Each capability set contains six sub-capability types.

**`AgentCapabilities` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `capability_id` | `String` | Unique UUID for audit correlation |
| `filesystem` | `FilesystemCapabilities` | File read/write/delete access |
| `tools` | `ToolCapabilities` | Tool category and name access |
| `network` | `NetworkCapabilities` | Domain and rate-limit controls |
| `spawning` | `SpawningCapabilities` | Child agent spawning limits |
| `git` | `GitCapabilities` | Git operation access |
| `quotas` | `ResourceQuotas` | Execution time, memory, token limits |

**Key methods:**

| Method | Description |
|--------|-------------|
| `allows_tool(name)` | Check if a tool name is permitted |
| `allows_read(path)` | Check if a file path is readable |
| `allows_write(path)` | Check if a file path is writable |
| `allows_domain(domain)` | Check if a network domain is permitted |
| `allows_git_op(op)` | Check if a git operation is permitted |
| `can_spawn_agent()` | Check if agent can spawn children |
| `derive_child()` | Create constrained child capabilities (reduced depth) |
| `intersect(other)` | Merge two sets, taking the most restrictive option |

### Capability Profiles

Three built-in profiles provide secure defaults for common use cases.

| Profile | Description |
|---------|-------------|
| `ReadOnly` | Read all files (except secrets), search tools, read-only git — safe for untrusted agents |
| `StandardDev` | Write to `src/`, `tests/`, `docs/`; most tools except code execution; common dev network domains; limited spawning |
| `FullAccess` | Complete access to all capabilities — for trusted orchestrators only |

```rust
use brainwires_permissions::CapabilityProfile;

let caps = CapabilityProfile::ReadOnly.build();
let caps = CapabilityProfile::StandardDev.build();
let caps = CapabilityProfile::FullAccess.build();
let caps = CapabilityProfile::Custom.build(); // empty, add capabilities manually
```

### Filesystem Capabilities

Granular control over file operations using glob patterns.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `read_paths` | `Vec<PathPattern>` | `[]` | Allowed read paths (glob) |
| `write_paths` | `Vec<PathPattern>` | `[]` | Allowed write paths (glob) |
| `denied_paths` | `Vec<PathPattern>` | `[]` | Denied paths override allows |
| `max_write_size` | `Option<u64>` | `None` | Maximum write size in bytes |
| `can_delete` | `bool` | `false` | Allow file deletion |
| `can_create_dirs` | `bool` | `false` | Allow directory creation |
| `follow_symlinks` | `bool` | `false` | Follow symbolic links |
| `access_hidden` | `bool` | `false` | Access hidden/dot files |

**`PathPattern`** — glob pattern matching (native) or string containment (WASM):

```rust
use brainwires_permissions::PathPattern;

let pattern = PathPattern::new("src/**/*.rs");
assert!(pattern.matches("src/lib.rs"));
assert!(pattern.matches("src/agents/mod.rs"));
assert!(!pattern.matches("target/debug/main"));
```

### Tool Capabilities

Tool access control by category and name.

**`ToolCategory` variants:**

| Category | Matched Tool Names |
|----------|-------------------|
| `FileRead` | `read_file`, `list_directory` |
| `FileWrite` | `write_file`, `edit_file` |
| `Search` | `query_codebase`, `search_*`, `grep_*` |
| `Git` | `git_*` (non-destructive) |
| `GitDestructive` | `git_reset`, `git_force_push`, `git_rebase` |
| `Bash` | `bash`, `execute_command` |
| `Web` | `web_fetch`, `web_search` |
| `CodeExecution` | `run_code`, `execute_*` |
| `AgentSpawn` | `agent_spawn` |
| `Planning` | `create_plan`, `update_plan` |
| `System` | `system_*`, `config_*` |

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `allowed_categories` | `Vec<ToolCategory>` | Permitted tool categories |
| `denied_tools` | `Vec<String>` | Explicit deny list (overrides categories) |
| `allowed_tools` | `Vec<String>` | Explicit allow list |
| `tools_requiring_approval` | `Vec<String>` | Tools that trigger approval requests |

### Network Capabilities

Domain-based access control with rate limiting.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allowed_domains` | `Vec<String>` | `[]` | Permitted domains (supports `*.github.com` wildcards) |
| `denied_domains` | `Vec<String>` | `[]` | Blocked domains (overrides allows) |
| `allow_all` | `bool` | `false` | Allow all network access |
| `rate_limit_per_minute` | `Option<u32>` | `None` | Maximum requests per minute |
| `allow_api_calls` | `bool` | `false` | Allow external API calls |
| `max_response_size` | `Option<u64>` | `None` | Maximum response size in bytes |

### Git Capabilities

Fine-grained git operation access with branch protection.

**`GitOperation` variants:** `Clone`, `Pull`, `Push`, `Commit`, `Branch`, `Checkout`, `Merge`, `Rebase`, `Tag`, `Stash`, `Reset`, `Diff`, `Log`, `Status`, `Fetch`, `ForcePush`

| Field | Type | Description |
|-------|------|-------------|
| `allowed_operations` | `Vec<GitOperation>` | Permitted git operations |
| `protected_branches` | `Vec<String>` | Branches that cannot be modified |
| `can_force_push` | `bool` | Allow force push (destructive) |
| `can_destructive` | `bool` | Allow destructive ops (reset, rebase, merge) |
| `require_pr` | `bool` | Require pull request for changes |

### Spawning Capabilities

Controls for hierarchical agent spawning.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `can_spawn` | `bool` | `false` | Agent can spawn children |
| `max_children` | `u32` | `0` | Maximum concurrent children |
| `max_depth` | `u32` | `0` | Maximum nesting depth |
| `can_elevate` | `bool` | `false` | Can request elevated permissions (requires approval) |

### Resource Quotas

Execution resource limits with three presets.

| Field | Type | Description |
|-------|------|-------------|
| `max_execution_time` | `Option<u64>` | Maximum execution time in seconds |
| `max_memory` | `Option<u64>` | Maximum memory in bytes |
| `max_tokens` | `Option<u64>` | Maximum token budget |
| `max_tool_calls` | `Option<u32>` | Maximum tool invocations |
| `max_files_modified` | `Option<u32>` | Maximum files modified |

**Presets:** `ResourceQuotas::conservative()`, `ResourceQuotas::standard()`, `ResourceQuotas::generous()`

### Policy Engine

Rule-based access control with priority-sorted evaluation.

```rust
use brainwires_permissions::{
    PolicyEngine, Policy, PolicyCondition, PolicyAction, EnforcementMode,
};

let mut engine = PolicyEngine::new();
engine.add_policy(Policy {
    name: "protect-secrets".into(),
    priority: 100,
    condition: PolicyCondition::FilePath("**/.env*".into()),
    action: PolicyAction::Deny,
    enforcement: EnforcementMode::Coercive,
    ..Default::default()
});
```

**`PolicyCondition` variants:**

| Condition | Description |
|-----------|-------------|
| `ToolName(pattern)` | Match tool by name |
| `ToolCategory(category)` | Match tool by category |
| `FilePath(pattern)` | Match file path with glob |
| `TrustLevel(min_level)` | Require minimum trust level |
| `Domain(pattern)` | Match network domain |
| `GitOperation(op)` | Match git operation |
| `TimeRange(start_hour, end_hour)` | Match time of day |
| `And(Vec<Condition>)` | All conditions must match |
| `Or(Vec<Condition>)` | Any condition must match |
| `Not(Box<Condition>)` | Negation |

**`PolicyAction` variants:**

| Action | Description |
|--------|-------------|
| `Allow` | Permit the request |
| `Deny` | Reject the request |
| `RequireApproval` | Pause and wait for human approval |
| `AllowWithAudit` | Allow but flag for audit review |
| `DenyWithMessage(msg)` | Deny with custom reason |
| `Escalate` | Escalate to parent agent or orchestrator |

**`EnforcementMode` variants:**

| Mode | Description |
|------|-------------|
| `Coercive` | Hard enforcement — cannot be overridden |
| `Normative` | Soft enforcement — can override with justification |
| `Adaptive` | Learns from overrides to adjust policy over time |

**`PolicyDecision` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `action` | `PolicyAction` | Resulting action |
| `matched_policy` | `Option<String>` | Name of matched policy |
| `reason` | `String` | Human-readable explanation |
| `should_audit` | `bool` | Whether to create audit event |

### Audit Logger

JSONL-based event logging with buffered writes, querying, and statistics.

**`AuditEventType` variants:**

| Type | Description |
|------|-------------|
| `ToolExecution` | Tool was executed |
| `FileAccess` | File was read or written |
| `NetworkRequest` | Network call was made |
| `AgentSpawn` | Child agent was spawned |
| `PolicyViolation` | Policy rule was violated |
| `TrustChange` | Trust score was modified |
| `HumanIntervention` | Human approved or rejected an action |
| `SessionStart` / `SessionEnd` | Session lifecycle |
| `ConfigChange` | Configuration was modified |
| `UserFeedback` | User submitted feedback |

**`ActionOutcome` variants:** `Success`, `Failure`, `Partial`, `Timeout`, `Cancelled`, `Denied`, `PendingApproval`, `Approved`, `Rejected`

**`AuditLogger` key methods:**

| Method | Description |
|--------|-------------|
| `log(event)` | Buffer an event (auto-flushes at 100 events) |
| `flush()` | Write buffered events to disk |
| `query(query)` | Filter events by agent, type, action, time range |
| `statistics()` | Compute aggregate stats (counts, rates) |
| `export_json()` / `export_csv()` | Export events in JSON or CSV format |
| `submit_feedback(signal)` | Record user feedback (thumbs up/down with optional correction) |

Important events (violations, trust changes, approvals, feedback) are flushed immediately.

**`AuditQuery` filters:**

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `Option<String>` | Filter by agent |
| `event_type` | `Option<AuditEventType>` | Filter by event type |
| `action` | `Option<String>` | Filter by action name |
| `outcome` | `Option<ActionOutcome>` | Filter by outcome |
| `start_time` / `end_time` | `Option<DateTime>` | Time range |
| `limit` | `Option<usize>` | Maximum results |

### Trust Manager

Dynamic reputation scoring for agents with violation-based penalties.

**`TrustLevel` (5 levels, ordered):**

| Level | Value | Score Range | Description |
|-------|-------|-------------|-------------|
| `Untrusted` | 0 | < 0.4 | New or compromised agents |
| `Low` | 1 | 0.4 – 0.7 | Limited track record |
| `Medium` | 2 | 0.7 – 0.9 | Established agents |
| `High` | 3 | ≥ 0.9 | Highly reliable agents |
| `System` | 4 | 1.0 (fixed) | Internal system agents |

**Violation penalties (exponential):**

| Severity | Base Penalty | Recent Penalty (24h) |
|----------|-------------|---------------------|
| Minor | 0.02 | 0.04 |
| Major | 0.08 | 0.15 |
| Critical | 0.15 | 0.30 |

Recent violations within 24 hours apply the higher penalty. Violations decay after 24 hours.

**`TrustManager` key methods:**

| Method | Description |
|--------|-------------|
| `record_success(agent_id)` | Increment success counter, recalculate score |
| `record_failure(agent_id)` | Increment total counter, recalculate score |
| `record_violation(agent_id, severity)` | Apply violation penalty |
| `get_trust_level(agent_id)` | Get current trust level |
| `set_trust_level(agent_id, level)` | Override trust level |
| `register_system_agent(agent_id)` | Set agent to System trust (score 1.0) |
| `statistics()` | Agent counts by level, total violations, average score |

Persisted to `~/.brainwires/trust_store.json`.

### Anomaly Detector

Real-time anomaly detection using per-agent sliding-window counters.

**Window semantics:** each threshold uses an event-timestamp window (not a
time-bucketed histogram). Every recorded event carries its `now_secs`
timestamp; on each `record_and_count` the counter evicts entries older than
`now - window_secs` and returns the in-window count. Counters are keyed per
`(agent_id, kind)`, so one noisy agent does not trip thresholds for its
peers, and a quiet agent retains no memory after its window elapses. There is
no internal bucket size — resolution is per-event. All three window durations
and thresholds are live-configurable via `AnomalyConfig`.

**`AnomalyKind` (4 types):**

| Kind | Trigger | Description |
|------|---------|-------------|
| `RepeatedPolicyViolation` | Same agent violates > threshold times in window | Policy abuse detection |
| `HighFrequencyToolCalls` | Agent calls tools faster than threshold | Runaway agent detection |
| `UnusualFileScopeRequest` | Tool access outside expected path prefixes | Scope creep detection |
| `RapidTrustChange` | Trust level changes > threshold times in window | Instability detection |

**`AnomalyConfig` defaults:**

| Field | Default | Description |
|-------|---------|-------------|
| `violation_threshold` | 3 | Max violations before flagged |
| `violation_window_secs` | 60 | Sliding window for violations |
| `tool_call_threshold` | 20 | Max tool calls in window |
| `tool_call_window_secs` | 10 | Sliding window for tool calls |
| `trust_change_threshold` | 3 | Max trust changes in window |
| `trust_change_window_secs` | 60 | Sliding window for trust changes |
| `expected_path_prefixes` | `None` | Optional path scope whitelist |

The anomaly detector is integrated into the audit logger — every logged event is automatically fed to the detector.

### Approval System

Async approval requests with severity classification.

**`ApprovalAction` variants:**

| Action | Severity | Description |
|--------|----------|-------------|
| `CreateDirectory` | Low | Create a new directory |
| `NetworkAccess` | Low | Make a network request |
| `WriteFile` | Medium | Write to a file |
| `EditFile` | Medium | Edit an existing file |
| `GitModify` | Medium | Modify git state |
| `Other(desc)` | Medium | Custom action |
| `DeleteFile` | High | Delete a file |
| `ExecuteCommand` | High | Execute a shell command |

**`ApprovalResponse` variants:**

| Response | Scope | Description |
|----------|-------|-------------|
| `Approve` | Single use | Allow this one request |
| `Deny` | Single use | Reject this one request |
| `ApproveForSession` | Session | Allow all similar requests this session |
| `DenyForSession` | Session | Deny all similar requests this session |

Approval requests use a `tokio::sync::oneshot` channel for async response delivery.

### Configuration

TOML-based configuration loaded from `~/.brainwires/permissions.toml`.

```toml
[default]
profile = "standard_dev"

[filesystem]
read_paths = ["**/*.rs", "**/*.toml", "**/*.md"]
write_paths = ["src/**", "tests/**"]
denied_paths = ["**/.env*", "**/secrets/**"]
max_write_size = "1MB"
can_delete = false

[tools]
allowed_categories = ["FileRead", "FileWrite", "Search", "Git"]
denied_tools = ["bash"]
always_approve = ["delete_file"]

[network]
allowed_domains = ["api.github.com", "*.crates.io"]
rate_limit = 30

[spawning]
enabled = true
max_children = 3
max_depth = 2

[git]
allowed_ops = ["commit", "push", "pull", "branch", "status", "diff", "log"]
protected_branches = ["main", "production"]
can_force_push = false

[quotas]
max_execution_time = "30m"
max_tool_calls = 200
max_files_modified = 20

[[policies.rules]]
name = "require-approval-for-destructive-git"
condition = { git_op = "Reset" }
action = "require_approval"
enforcement = "coercive"
```

**Parsing utilities:**

| Function | Input Examples | Description |
|----------|---------------|-------------|
| `parse_size(s)` | `"1MB"`, `"512KB"`, `"1GB"` | Parse human-readable size to bytes |
| `parse_duration(s)` | `"30m"`, `"1h"`, `"90s"` | Parse human-readable duration to seconds |
| `parse_tool_category(s)` | `"FileRead"`, `"file_read"` | Case-insensitive category parsing |
| `parse_git_operation(s)` | `"ForcePush"`, `"force_push"` | Case-insensitive git op parsing |

**Helper functions:**

| Function | Description |
|----------|-------------|
| `default_permissions_path()` | Returns `~/.brainwires/permissions.toml` |
| `ensure_permissions_dir()` | Creates `~/.brainwires/` if it doesn't exist |
| `PermissionsConfig::load_or_default(path)` | Load config file with fallback to defaults |

## Usage Examples

### Capability Profiles and Child Derivation

```rust
use brainwires_permissions::{AgentCapabilities, CapabilityProfile};

// Parent agent with standard dev capabilities
let parent_caps = CapabilityProfile::StandardDev.build();
assert!(parent_caps.allows_tool("write_file"));
assert!(parent_caps.allows_write("src/main.rs"));

// Derive constrained capabilities for a child agent
let child_caps = parent_caps.derive_child();
assert!(child_caps.allows_tool("write_file")); // inherits tool access
// Child has reduced spawning depth

// Intersect two capability sets (most restrictive wins)
let restricted = parent_caps.intersect(&CapabilityProfile::ReadOnly.build());
assert!(!restricted.allows_write("src/main.rs")); // read-only wins
```

### Policy Engine with Custom Rules

```rust
use brainwires_permissions::{
    PolicyEngine, Policy, PolicyCondition, PolicyAction, EnforcementMode,
    PolicyRequest, ToolCategory,
};

let mut engine = PolicyEngine::with_defaults();

// Add custom policy: deny bash during off-hours
engine.add_policy(Policy {
    name: "no-bash-at-night".into(),
    priority: 50,
    condition: PolicyCondition::And(vec![
        PolicyCondition::ToolCategory(ToolCategory::Bash),
        PolicyCondition::TimeRange(22, 6),
    ]),
    action: PolicyAction::DenyWithMessage("Bash disabled outside business hours".into()),
    enforcement: EnforcementMode::Coercive,
    ..Default::default()
});

let request = PolicyRequest {
    tool_name: Some("bash".into()),
    tool_category: Some(ToolCategory::Bash),
    ..Default::default()
};
let decision = engine.evaluate(&request);
```

### Audit Logging and Querying

```rust
use brainwires_permissions::{
    AuditLogger, AuditEvent, AuditEventType, ActionOutcome, AuditQuery,
};

let logger = AuditLogger::new()?;

// Log a tool execution
let event = AuditEvent {
    event_type: AuditEventType::ToolExecution,
    agent_id: "agent-1".into(),
    action: "write_file".into(),
    target: Some("src/lib.rs".into()),
    outcome: ActionOutcome::Success,
    ..Default::default()
};
logger.log(event);

// Query recent violations
let violations = logger.query(AuditQuery {
    event_type: Some(AuditEventType::PolicyViolation),
    limit: Some(10),
    ..Default::default()
});

// Export for review
let json = logger.export_json()?;
let csv = logger.export_csv()?;

// Get aggregate statistics
let stats = logger.statistics();
```

### Trust Management

```rust
use brainwires_permissions::{TrustManager, TrustLevel, ViolationSeverity};

let mut manager = TrustManager::new()?;

// Record successful operations to build trust
manager.record_success("agent-1");
manager.record_success("agent-1");
manager.record_success("agent-1");

let level = manager.get_trust_level("agent-1");
// Level increases as success rate improves

// Record a violation (reduces trust)
manager.record_violation("agent-1", ViolationSeverity::Major);
let level = manager.get_trust_level("agent-1");
// Level may drop after violation penalty applied

// Register a system-level agent (always trusted)
manager.register_system_agent("orchestrator");
assert_eq!(manager.get_trust_level("orchestrator"), TrustLevel::System);

// View statistics
let stats = manager.statistics();
```

### Anomaly Detection

```rust
use brainwires_permissions::{AnomalyDetector, AnomalyConfig};

let config = AnomalyConfig {
    violation_threshold: 3,
    violation_window_secs: 60,
    tool_call_threshold: 20,
    tool_call_window_secs: 10,
    expected_path_prefixes: Some(vec!["src/".into(), "tests/".into()]),
    ..Default::default()
};

let detector = AnomalyDetector::new(config);

// Anomaly events are automatically detected when integrated with AuditLogger
// Drain pending anomalies
let anomalies = detector.drain_events();
for anomaly in &anomalies {
    println!("Anomaly: {:?} for agent {}", anomaly.kind, anomaly.agent_id);
}
```

### Loading Configuration from TOML

```rust
use brainwires_permissions::{PermissionsConfig, default_permissions_path};

// Load from default path (~/.brainwires/permissions.toml)
let config = PermissionsConfig::load_or_default(default_permissions_path());

// Build capabilities from config
let caps = config.build_capabilities();

// Build policy engine from config
let engine = config.build_policy_engine();
```

## File Locations

| File | Description |
|------|-------------|
| `~/.brainwires/permissions.toml` | Permission configuration |
| `~/.brainwires/audit/audit.jsonl` | Audit event log |
| `~/.brainwires/trust_store.json` | Persistent trust scores |

## Integration with Brainwires

Use via the `brainwires` facade crate:

```toml
[dependencies]
brainwires = "0.11"
```

Or use standalone — `brainwires-permission` depends only on `brainwires-core`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
