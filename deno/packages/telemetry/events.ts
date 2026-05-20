/**
 * Typed analytics event variants emitted anywhere in the framework.
 *
 * Equivalent to Rust's `brainwires_telemetry::events::AnalyticsEvent`.
 */

/** Compliance metadata attached to auditable events. */
export interface ComplianceMetadata {
  /** ISO 3166-1 alpha-2 (e.g. "EU", "US"). */
  data_region?: string;
  pii_present?: boolean;
  retention_days?: number;
  /** "GDPR" | "HIPAA" | "EU_AI_ACT" | … */
  regulation?: string;
  audit_required?: boolean;
}

/** A typed analytics event. `timestamp` is ISO 8601. */
export type AnalyticsEvent =
  | {
    event_type: "provider_call";
    session_id: string | null;
    provider: string;
    model: string;
    prompt_tokens: number;
    completion_tokens: number;
    duration_ms: number;
    cost_usd: number;
    success: boolean;
    timestamp: string;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
    compliance?: ComplianceMetadata;
  }
  | {
    event_type: "agent_run";
    session_id: string | null;
    agent_id: string;
    task_id: string;
    prompt_hash: string;
    success: boolean;
    total_iterations: number;
    total_tool_calls: number;
    tool_error_count: number;
    tools_used: string[];
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost_usd: number;
    duration_ms: number;
    failure_category: string | null;
    timestamp: string;
    compliance?: ComplianceMetadata;
  }
  | {
    event_type: "tool_call";
    session_id: string | null;
    agent_id: string | null;
    tool_name: string;
    tool_use_id: string;
    is_error: boolean;
    duration_ms: number | null;
    timestamp: string;
  }
  | {
    event_type: "mcp_request";
    session_id: string | null;
    server_name: string;
    tool_name: string;
    success: boolean;
    duration_ms: number;
    timestamp: string;
  }
  | {
    event_type: "channel_message";
    session_id: string | null;
    channel_type: string;
    /** "inbound" | "outbound". */
    direction: string;
    /** Length in bytes after PII scrubbing. */
    message_len: number;
    timestamp: string;
  }
  | {
    event_type: "storage_op";
    session_id: string | null;
    store_type: string;
    operation: string;
    success: boolean;
    duration_ms: number;
    timestamp: string;
  }
  | {
    event_type: "network_message";
    session_id: string | null;
    protocol: string;
    direction: string;
    bytes: number;
    success: boolean;
    timestamp: string;
  }
  | {
    event_type: "dream_cycle";
    session_id: string | null;
    sessions_processed: number;
    messages_summarized: number;
    facts_extracted: number;
    tokens_before: number;
    tokens_after: number;
    duration_ms: number;
    timestamp: string;
  }
  | {
    event_type: "autonomy_session";
    session_id: string | null;
    tasks_attempted: number;
    tasks_succeeded: number;
    tasks_failed: number;
    total_cost_usd: number;
    duration_ms: number;
    timestamp: string;
  }
  | {
    event_type: "custom";
    session_id: string | null;
    name: string;
    payload: unknown;
    timestamp: string;
  };

/** Event type tag. */
export function eventType(e: AnalyticsEvent): AnalyticsEvent["event_type"] {
  return e.event_type;
}

/** ISO 8601 timestamp. */
export function eventTimestamp(e: AnalyticsEvent): string {
  return e.timestamp;
}

/** Session id carried by the event, if any. */
export function eventSessionId(e: AnalyticsEvent): string | null {
  return e.session_id ?? null;
}
