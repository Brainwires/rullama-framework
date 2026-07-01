/**
 * Audit System — Comprehensive logging for security and compliance.
 *
 * Provides audit logging for all permission-related events including tool
 * executions, file access, network requests, policy evaluations, trust
 * level changes, and human interventions.
 *
 * Rust equivalent: `rullama-permissions/src/audit.rs`
 * @module
 */

import type { PolicyDecision } from "./policy.ts";

// Anomaly detection lives in `@rullama/telemetry`. AuditLogger no longer
// wires it up internally — callers observe events with their own detector:
//
//   import { AnomalyDetector, defaultAnomalyConfig } from "@rullama/telemetry";
//   const detector = new AnomalyDetector(defaultAnomalyConfig());
//   auditLogger.log(event);
//   detector.observe(event);

// ── Audit Event Type ────────────────────────────────────────────────

/**
 * Type of audit event.
 *
 * Rust equivalent: `AuditEventType` enum (serde `rename_all = "snake_case"`)
 */
export type AuditEventType =
  | "tool_execution"
  | "file_access"
  | "network_request"
  | "agent_spawn"
  | "policy_violation"
  | "trust_change"
  | "human_intervention"
  | "session_start"
  | "session_end"
  | "config_change"
  | "user_feedback";

// ── Action Outcome ──────────────────────────────────────────────────

/**
 * Outcome of an action.
 *
 * Rust equivalent: `ActionOutcome` enum (serde `rename_all = "snake_case"`)
 */
export type ActionOutcome =
  | "success"
  | "failure"
  | "partial"
  | "timeout"
  | "cancelled"
  | "denied"
  | "pending_approval"
  | "approved"
  | "rejected";

// ── Feedback ────────────────────────────────────────────────────────

/**
 * Polarity of a user feedback signal.
 *
 * Rust equivalent: `FeedbackPolarity` enum (serde `rename_all = "snake_case"`)
 */
export type FeedbackPolarity = "thumbs_up" | "thumbs_down";

/**
 * A user feedback signal associated with a single agent run.
 *
 * Rust equivalent: `FeedbackSignal` struct
 */
export interface FeedbackSignal {
  /** Unique identifier for this feedback submission. */
  id: string;
  /** The run UUID this feedback is associated with. */
  run_id: string;
  /** Whether the user approved or disapproved the output. */
  polarity: FeedbackPolarity;
  /** Optional free-text correction or comment from the user. */
  correction: string | undefined;
  /** When the feedback was submitted. */
  submitted_at: string;
}

// ── Audit Event ─────────────────────────────────────────────────────

/**
 * A single audit event.
 *
 * Rust equivalent: `AuditEvent` struct
 */
export interface AuditEvent {
  /** Unique event ID. */
  id: string;
  /** When the event occurred (ISO 8601). */
  timestamp: string;
  /** Type of event. */
  event_type: AuditEventType;
  /** Agent that triggered the event. */
  agent_id: string | undefined;
  /** Action that was performed. */
  action: string;
  /** Target of the action (file path, domain, etc.). */
  target: string | undefined;
  /** Policy that was evaluated. */
  policy_id: string | undefined;
  /** Decision made by policy engine. */
  decision: string | undefined;
  /** Trust level at time of event. */
  trust_level: number | undefined;
  /** Outcome of the action. */
  outcome: ActionOutcome;
  /** Duration in milliseconds (if applicable). */
  duration_ms: number | undefined;
  /** Error message (if failed). */
  error: string | undefined;
  /** Additional metadata. */
  metadata: Record<string, string>;
}

/**
 * Create a new audit event.
 *
 * Rust equivalent: `AuditEvent::new()`
 */
export function createAuditEvent(eventType: AuditEventType): AuditEvent {
  return {
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    event_type: eventType,
    agent_id: undefined,
    action: "",
    target: undefined,
    policy_id: undefined,
    decision: undefined,
    trust_level: undefined,
    outcome: "success",
    duration_ms: undefined,
    error: undefined,
    metadata: {},
  };
}

/** Builder-style helpers for AuditEvent (mutates and returns the event). */
export function withAgent(event: AuditEvent, agentId: string): AuditEvent {
  event.agent_id = agentId;
  return event;
}
export function withAction(event: AuditEvent, action: string): AuditEvent {
  event.action = action;
  return event;
}
export function withTarget(event: AuditEvent, target: string): AuditEvent {
  event.target = target;
  return event;
}
export function withPolicyDecision(
  event: AuditEvent,
  decision: PolicyDecision,
): AuditEvent {
  event.policy_id = decision.matched_policy;
  event.decision = decision.reason;
  return event;
}
export function withTrustLevel(event: AuditEvent, level: number): AuditEvent {
  event.trust_level = level;
  return event;
}
export function withOutcome(
  event: AuditEvent,
  outcome: ActionOutcome,
): AuditEvent {
  event.outcome = outcome;
  return event;
}
export function withDuration(
  event: AuditEvent,
  durationMs: number,
): AuditEvent {
  event.duration_ms = durationMs;
  return event;
}
export function withError(event: AuditEvent, error: string): AuditEvent {
  event.error = error;
  event.outcome = "failure";
  return event;
}
export function withMetadata(
  event: AuditEvent,
  key: string,
  value: string,
): AuditEvent {
  event.metadata[key] = value;
  return event;
}

// ── Audit Query ─────────────────────────────────────────────────────

/**
 * Query parameters for searching audit logs.
 *
 * Rust equivalent: `AuditQuery` struct
 */
export interface AuditQuery {
  /** Filter by agent ID. */
  agent_id: string | undefined;
  /** Filter by event type. */
  event_type: AuditEventType | undefined;
  /** Filter by action. */
  action: string | undefined;
  /** Filter by outcome. */
  outcome: ActionOutcome | undefined;
  /** Filter events after this time (ISO 8601). */
  since: string | undefined;
  /** Filter events before this time (ISO 8601). */
  until: string | undefined;
  /** Maximum number of results. */
  limit: number | undefined;
}

/**
 * Create a new empty AuditQuery.
 *
 * Rust equivalent: `AuditQuery::new()`
 */
export function createAuditQuery(overrides?: Partial<AuditQuery>): AuditQuery {
  return {
    agent_id: undefined,
    event_type: undefined,
    action: undefined,
    outcome: undefined,
    since: undefined,
    until: undefined,
    limit: undefined,
    ...overrides,
  };
}

/**
 * Check if an event matches this query.
 *
 * Rust equivalent: `AuditQuery::matches()`
 */
export function queryMatches(query: AuditQuery, event: AuditEvent): boolean {
  if (query.agent_id !== undefined && event.agent_id !== query.agent_id) {
    return false;
  }
  if (query.event_type !== undefined && event.event_type !== query.event_type) {
    return false;
  }
  if (query.action !== undefined && !event.action.includes(query.action)) {
    return false;
  }
  if (query.outcome !== undefined && event.outcome !== query.outcome) {
    return false;
  }
  if (query.since !== undefined && event.timestamp < query.since) return false;
  if (query.until !== undefined && event.timestamp > query.until) return false;
  return true;
}

// ── Audit Statistics ────────────────────────────────────────────────

/**
 * Audit statistics.
 *
 * Rust equivalent: `AuditStatistics` struct
 */
export interface AuditStatistics {
  /** Total number of audit events. */
  total_events: number;
  /** Number of tool executions. */
  tool_executions: number;
  /** Number of policy violations. */
  policy_violations: number;
  /** Number of human interventions. */
  human_interventions: number;
  /** Number of successful actions. */
  successful_actions: number;
  /** Number of denied actions. */
  denied_actions: number;
  /** Number of failed actions. */
  failed_actions: number;
}

// ── Audit Logger ────────────────────────────────────────────────────

const DEFAULT_AUDIT_BUFFER_SIZE = 100;

/** Important event types that should be written immediately. */
const IMPORTANT_EVENT_TYPES: ReadonlySet<AuditEventType> = new Set([
  "policy_violation",
  "trust_change",
  "human_intervention",
  "user_feedback",
]);

/**
 * Audit logger for recording permission events.
 *
 * Writes JSONL to disk. Buffers non-critical events and flushes when
 * the buffer is full or on explicit flush/dispose.
 *
 * Rust equivalent: `AuditLogger` struct
 */
export class AuditLogger {
  #logPath: string;
  #buffer: AuditEvent[] = [];
  #maxBufferSize: number;
  #enabled = true;

  private constructor(
    logPath: string,
    maxBufferSize = DEFAULT_AUDIT_BUFFER_SIZE,
  ) {
    this.#logPath = logPath;
    this.#maxBufferSize = maxBufferSize;
  }

  /**
   * Create an audit logger with a custom path.
   *
   * Rust equivalent: `AuditLogger::with_path()`
   */
  static withPath(path: string): AuditLogger {
    // Ensure parent directory exists
    const parent = path.substring(0, path.lastIndexOf("/"));
    if (parent) {
      try {
        Deno.mkdirSync(parent, { recursive: true });
      } catch { /* ignore */ }
    }
    return new AuditLogger(path);
  }

  /**
   * Create a new audit logger with the default path (~/.rullama/audit/audit.jsonl).
   *
   * Rust equivalent: `AuditLogger::new()`
   */
  static create(): AuditLogger {
    const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE") ?? ".";
    const logDir = `${home}/.rullama/audit`;
    try {
      Deno.mkdirSync(logDir, { recursive: true });
    } catch { /* ignore */ }
    return new AuditLogger(`${logDir}/audit.jsonl`);
  }

  /** Enable or disable logging. Rust equivalent: `AuditLogger::set_enabled()` */
  setEnabled(enabled: boolean): void {
    this.#enabled = enabled;
  }

  /**
   * Log an audit event.
   *
   * Rust equivalent: `AuditLogger::log()`
   */
  log(event: AuditEvent): void {
    if (!this.#enabled) return;

    if (IMPORTANT_EVENT_TYPES.has(event.event_type)) {
      this.#writeEvent(event);
    } else {
      this.#buffer.push(event);
      if (this.#buffer.length >= this.#maxBufferSize) {
        this.#flushBuffer();
      }
    }
  }

  /**
   * Log a tool execution.
   *
   * Rust equivalent: `AuditLogger::log_tool_execution()`
   */
  logToolExecution(
    agentId: string | undefined,
    toolName: string,
    target: string | undefined,
    outcome: ActionOutcome,
    durationMs?: number,
  ): void {
    let event = createAuditEvent("tool_execution");
    event = withAction(event, toolName);
    event = withOutcome(event, outcome);
    if (agentId) event = withAgent(event, agentId);
    if (target) event = withTarget(event, target);
    if (durationMs !== undefined) event = withDuration(event, durationMs);
    this.log(event);
  }

  /**
   * Log a denied action.
   *
   * Rust equivalent: `AuditLogger::log_denied()`
   */
  logDenied(
    agentId: string | undefined,
    action: string,
    target: string | undefined,
    reason: string,
  ): void {
    let event = createAuditEvent("policy_violation");
    event = withAction(event, action);
    event = withOutcome(event, "denied");
    event = withMetadata(event, "reason", reason);
    if (agentId) event = withAgent(event, agentId);
    if (target) event = withTarget(event, target);
    this.log(event);
  }

  /**
   * Log a human approval.
   *
   * Rust equivalent: `AuditLogger::log_approval()`
   */
  logApproval(
    agentId: string | undefined,
    action: string,
    approved: boolean,
    justification?: string,
  ): void {
    let event = createAuditEvent("human_intervention");
    event = withAction(event, action);
    event = withOutcome(event, approved ? "approved" : "rejected");
    if (agentId) event = withAgent(event, agentId);
    if (justification) {
      event = withMetadata(event, "justification", justification);
    }
    this.log(event);
  }

  /**
   * Log a trust level change.
   *
   * Rust equivalent: `AuditLogger::log_trust_change()`
   */
  logTrustChange(
    agentId: string,
    oldLevel: number,
    newLevel: number,
    reason: string,
  ): void {
    let event = createAuditEvent("trust_change");
    event = withAgent(event, agentId);
    event = withAction(event, "trust_change");
    event = withTrustLevel(event, newLevel);
    event = withMetadata(event, "old_level", String(oldLevel));
    event = withMetadata(event, "reason", reason);
    this.log(event);
  }

  /**
   * Submit user feedback for a specific agent run.
   *
   * Rust equivalent: `AuditLogger::submit_feedback()`
   */
  submitFeedback(
    runId: string,
    polarity: FeedbackPolarity,
    correction?: string,
  ): FeedbackSignal {
    const signal: FeedbackSignal = {
      id: crypto.randomUUID(),
      run_id: runId,
      polarity,
      correction,
      submitted_at: new Date().toISOString(),
    };

    let event = createAuditEvent("user_feedback");
    event = withAction(event, "user_feedback");
    event = withMetadata(event, "run_id", runId);
    event = withMetadata(event, "polarity", polarity);
    event = withMetadata(event, "feedback_id", signal.id);
    event = withOutcome(event, "success");
    if (correction) event = withMetadata(event, "correction", correction);

    this.log(event);
    return signal;
  }

  /**
   * Query all feedback signals associated with a specific run ID.
   *
   * Rust equivalent: `AuditLogger::get_feedback_for_run()`
   */
  getFeedbackForRun(runId: string): FeedbackSignal[] {
    const events = this.query(
      createAuditQuery({ event_type: "user_feedback" }),
    );
    const results: FeedbackSignal[] = [];
    for (const e of events) {
      if (e.metadata["run_id"] !== runId) continue;
      const feedbackId = e.metadata["feedback_id"];
      const polarityStr = e.metadata["polarity"];
      if (!feedbackId || !polarityStr) continue;
      if (polarityStr !== "thumbs_up" && polarityStr !== "thumbs_down") {
        continue;
      }
      results.push({
        id: feedbackId,
        run_id: e.metadata["run_id"],
        polarity: polarityStr as FeedbackPolarity,
        correction: e.metadata["correction"],
        submitted_at: e.timestamp,
      });
    }
    return results;
  }

  /** Flush the buffer to disk. Rust equivalent: `AuditLogger::flush()` */
  flush(): void {
    this.#flushBuffer();
  }

  /**
   * Query audit events.
   *
   * Rust equivalent: `AuditLogger::query()`
   */
  query(query: AuditQuery): AuditEvent[] {
    const results: AuditEvent[] = [];

    // Check buffer
    for (const event of this.#buffer) {
      if (queryMatches(query, event)) results.push(event);
    }

    // Read from file
    try {
      const content = Deno.readTextFileSync(this.#logPath);
      for (const line of content.split("\n")) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line) as AuditEvent;
          if (queryMatches(query, event)) results.push(event);
        } catch { /* skip malformed lines */ }
      }
    } catch { /* file doesn't exist yet */ }

    // Sort by timestamp (newest first)
    results.sort((a, b) => b.timestamp.localeCompare(a.timestamp));

    // Apply limit
    if (query.limit !== undefined) {
      results.splice(query.limit);
    }

    return results;
  }

  /** Get recent events. Rust equivalent: `AuditLogger::recent()` */
  recent(count: number): AuditEvent[] {
    return this.query(createAuditQuery({ limit: count }));
  }

  /** Count events matching a query. Rust equivalent: `AuditLogger::count()` */
  count(query: AuditQuery): number {
    return this.query(query).length;
  }

  /** Export audit log to JSON. Rust equivalent: `AuditLogger::export_json()` */
  exportJson(query: AuditQuery): string {
    return JSON.stringify(this.query(query), null, 2);
  }

  /** Export audit log to CSV. Rust equivalent: `AuditLogger::export_csv()` */
  exportCsv(query: AuditQuery): string {
    const events = this.query(query);
    let csv = "timestamp,event_type,agent_id,action,target,outcome,policy_id\n";
    for (const event of events) {
      csv += `${event.timestamp},${event.event_type},${
        event.agent_id ?? ""
      },${event.action},${event.target ?? ""},${event.outcome},${
        event.policy_id ?? ""
      }\n`;
    }
    return csv;
  }

  /**
   * Get audit statistics.
   *
   * Rust equivalent: `AuditLogger::statistics()`
   */
  statistics(since?: string): AuditStatistics {
    const query = createAuditQuery(since ? { since } : {});
    const events = this.query(query);

    const stats: AuditStatistics = {
      total_events: events.length,
      tool_executions: 0,
      policy_violations: 0,
      human_interventions: 0,
      successful_actions: 0,
      denied_actions: 0,
      failed_actions: 0,
    };

    for (const event of events) {
      if (event.event_type === "tool_execution") stats.tool_executions++;
      if (event.event_type === "policy_violation") stats.policy_violations++;
      if (event.event_type === "human_intervention") {
        stats.human_interventions++;
      }
      if (event.outcome === "success") stats.successful_actions++;
      if (event.outcome === "denied") stats.denied_actions++;
      if (event.outcome === "failure") stats.failed_actions++;
    }

    return stats;
  }

  // ── Private helpers ─────────────────────────────────────────────

  #writeEvent(event: AuditEvent): void {
    try {
      const json = JSON.stringify(event);
      Deno.writeTextFileSync(this.#logPath, json + "\n", {
        append: true,
        create: true,
      });
    } catch { /* ignore write errors */ }
  }

  #flushBuffer(): void {
    if (this.#buffer.length === 0) return;
    try {
      const lines = this.#buffer.map((e) => JSON.stringify(e)).join("\n") +
        "\n";
      Deno.writeTextFileSync(this.#logPath, lines, {
        append: true,
        create: true,
      });
    } catch { /* ignore write errors */ }
    this.#buffer = [];
  }
}
