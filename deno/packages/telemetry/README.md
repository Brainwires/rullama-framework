# @rullama/telemetry

Analytics events, sinks, outcome metrics, billing hooks and anomaly detection.
Depends on `@rullama/core` only.

## What you get

- **`AnalyticsEvent`** — 10 typed event variants (`provider_call`, `agent_run`,
  `tool_call`, `mcp_request`, `channel_message`, `storage_op`,
  `network_message`, `dream_cycle`, `autonomy_session`, `custom`).
- **`UsageEvent`** — billable-action payload (`tokens`, `tool_call`,
  `sandbox_seconds`, `api_call`, `custom`).
- **`AnalyticsCollector`** — fan-out to multiple sinks + optional `onEvent`
  callback for OTLP / custom piping.
- **`MemoryAnalyticsSink`** — ring-buffer sink for tests.
- **`MetricsRegistry`** — outcome aggregation with Prometheus text exposition.
- **`BillingHook`** interface — advisory `onUsage` + enforced `authorize`.
- **`AnomalyDetector`** — frequency / behaviour-change detection over
  `ObservedEvent`s (`observe` / `drainAnomalies`).
- PII helpers — `hashSessionId`, `redactSecrets`.

## Install

```sh
deno add jsr:@rullama/telemetry
```

## Example

```ts
import {
  AnalyticsCollector,
  AnomalyDetector,
  defaultAnomalyConfig,
  MemoryAnalyticsSink,
  MetricsRegistry,
} from "@rullama/telemetry";

const memory = new MemoryAnalyticsSink(1024);
const metrics = new MetricsRegistry();
const collector = new AnalyticsCollector([memory, metrics]);

// OTLP / logger passthrough, if you want one:
collector.onEvent((e) => console.debug(e.event_type));

collector.record({
  event_type: "agent_run",
  session_id: null,
  agent_id: "code-review",
  task_id: "t-42",
  prompt_hash: "abc",
  success: true,
  total_iterations: 3,
  total_tool_calls: 5,
  tool_error_count: 0,
  tools_used: ["read_file", "search_code"],
  total_prompt_tokens: 1200,
  total_completion_tokens: 300,
  total_cost_usd: 0.015,
  duration_ms: 4200,
  failure_category: null,
  timestamp: new Date().toISOString(),
});
await collector.flush();

const body = metrics.prometheusText();
// wire `body` to your /metrics HTTP handler
console.log(body.split("\n").length, memory.len());

// Anomaly detection over an event stream (used by @rullama/permission audits)
const detector = new AnomalyDetector(defaultAnomalyConfig());
detector.observe({
  timestamp: new Date().toISOString(),
  event_type: "policy_violation",
  agent_id: "code-review",
});
console.log(detector.drainAnomalies().length);
```

## What's intentionally not ported

- **SQLite sink + SQL query layer** — use your own `AnalyticsSink` against Deno
  KV, Postgres, or OTLP directly. Most production deployments will want OTLP
  anyway.
- **tracing-crate layer** — Deno has no `tracing` equivalent. Use
  `collector.onEvent(cb)` to pipe events to your logger / exporter.
- **Heavy PII detectors** — we ship `hashSessionId` and `redactSecrets`. Email /
  phone / SSN detection lives Rust-side until explicitly requested.

## Equivalent Rust crate

`rullama-telemetry` — same event shape, same semantics. Counters are plain
numbers instead of atomics because JS isolates are single-threaded.
