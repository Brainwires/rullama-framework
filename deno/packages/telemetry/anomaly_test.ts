/**
 * Tests for anomaly.ts — mirrors Rust tests in anomaly.rs
 *
 * Note: this test pulls audit-event construction helpers from
 * `@rullama/permissions` (post-Phase-9 rename: `@rullama/permission`)
 * to exercise the detector against the same shape used in production. The
 * detector itself has no dependency on the permission package.
 */
import { assert, assertEquals } from "@std/assert";
import { AnomalyDetector, defaultAnomalyConfig } from "./anomaly.ts";
import {
  type AuditEventType,
  createAuditEvent,
  withAction,
  withAgent,
  withOutcome,
  withTarget,
} from "@rullama/permission";

function makeEvent(eventType: AuditEventType, agent: string) {
  let event = createAuditEvent(eventType);
  event = withAgent(event, agent);
  event = withAction(event, "test_action");
  return event;
}

function makeEventWithTarget(
  eventType: AuditEventType,
  agent: string,
  target: string,
) {
  let event = createAuditEvent(eventType);
  event = withAgent(event, agent);
  event = withAction(event, "test_action");
  event = withTarget(event, target);
  event = withOutcome(event, "success");
  return event;
}

Deno.test("no anomaly below threshold", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    violation_threshold: 3,
  });
  const e = makeEvent("policy_violation", "agent-1");
  detector.observe(e);
  detector.observe(e);
  assertEquals(detector.pendingCount(), 0);
});

Deno.test("repeated violations trigger anomaly", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    violation_threshold: 3,
    violation_window_secs: 60,
  });
  const e = makeEvent("policy_violation", "agent-1");
  detector.observe(e);
  detector.observe(e);
  detector.observe(e);
  assertEquals(detector.pendingCount(), 1);
  const anomalies = detector.drainAnomalies();
  assertEquals(anomalies[0].kind.kind, "repeated_policy_violation");
});

Deno.test("high frequency tool calls", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    tool_call_threshold: 5,
    tool_call_window_secs: 60,
  });
  const e = makeEvent("tool_execution", "agent-2");
  for (let i = 0; i < 5; i++) {
    detector.observe(e);
  }
  assertEquals(detector.pendingCount(), 1);
  const anomalies = detector.drainAnomalies();
  assertEquals(anomalies[0].kind.kind, "high_frequency_tool_calls");
});

Deno.test("unusual file scope request", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    expected_path_prefixes: ["/workspace/"],
    tool_call_threshold: 1_000,
  });
  const e = makeEventWithTarget("tool_execution", "agent-3", "/etc/secrets");
  detector.observe(e);
  const anomalies = detector.drainAnomalies();
  assert(anomalies.some((a) =>
    a.kind.kind === "unusual_file_scope_request" &&
    (a.kind as { path: string }).path === "/etc/secrets"
  ));
});

Deno.test("within scope path - no scope anomaly", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    expected_path_prefixes: ["/workspace/"],
    tool_call_threshold: 1_000,
  });
  const e = makeEventWithTarget(
    "tool_execution",
    "agent-3",
    "/workspace/src/main.rs",
  );
  detector.observe(e);
  const anomalies = detector.drainAnomalies();
  assert(!anomalies.some((a) => a.kind.kind === "unusual_file_scope_request"));
});

Deno.test("rapid trust change", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    trust_change_threshold: 3,
    trust_change_window_secs: 60,
  });
  const e = makeEvent("trust_change", "agent-4");
  for (let i = 0; i < 3; i++) {
    detector.observe(e);
  }
  const anomalies = detector.drainAnomalies();
  assert(anomalies.some((a) => a.kind.kind === "rapid_trust_change"));
});

Deno.test("drain clears pending", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    violation_threshold: 1,
  });
  const e = makeEvent("policy_violation", "agent-5");
  detector.observe(e);
  assertEquals(detector.pendingCount(), 1);
  detector.drainAnomalies();
  assertEquals(detector.pendingCount(), 0);
});

Deno.test("different agents tracked separately", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    violation_threshold: 3,
  });
  const e1 = makeEvent("policy_violation", "agent-A");
  const e2 = makeEvent("policy_violation", "agent-B");
  detector.observe(e1);
  detector.observe(e1);
  detector.observe(e2);
  detector.observe(e2);
  assertEquals(detector.pendingCount(), 0);
});

Deno.test("anomaly event has agent_id", () => {
  const detector = new AnomalyDetector({
    ...defaultAnomalyConfig(),
    violation_threshold: 1,
  });
  const e = makeEvent("policy_violation", "my-agent");
  detector.observe(e);
  const anomalies = detector.drainAnomalies();
  assertEquals(anomalies[0].agent_id, "my-agent");
});
