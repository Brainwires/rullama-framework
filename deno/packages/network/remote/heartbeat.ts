/**
 * @module remote/heartbeat
 *
 * Heartbeat collector for agent telemetry and liveness detection.
 * Collects information about running agents and detects state changes.
 *
 * Equivalent to Rust's `rullama-network::remote::heartbeat`.
 */

import type { AgentEventType, RemoteAgentInfo } from "./protocol.ts";

// ============================================================================
// Heartbeat Data
// ============================================================================

/** Data collected during a heartbeat. */
export interface HeartbeatData {
  /** Current state of all agents. */
  agents: RemoteAgentInfo[];
  /** System CPU load (0.0 to 1.0). */
  system_load: number;
  /** Hostname of the machine. */
  hostname: string;
  /** Operating system. */
  os: string;
  /** Client version. */
  version: string;
}

// ============================================================================
// Agent Event
// ============================================================================

/** Agent state change event. */
export interface AgentEvent {
  /** Type of event. */
  event_type: AgentEventType;
  /** Agent session ID. */
  agent_id: string;
  /** Additional event data. */
  data: unknown;
}

// ============================================================================
// Agent Info Provider
// ============================================================================

/** Callback that provides current agent info. Injected by the host. */
export type AgentInfoProvider = () =>
  | Promise<RemoteAgentInfo[]>
  | RemoteAgentInfo[];

// ============================================================================
// Heartbeat Collector
// ============================================================================

/**
 * Collects heartbeat data and detects agent changes.
 *
 * Unlike the Rust version which reads IPC sockets from a sessions directory,
 * this TypeScript port accepts an `AgentInfoProvider` callback that the host
 * application supplies. This keeps the collector decoupled from any specific
 * IPC mechanism.
 */
export class HeartbeatCollector {
  /** Last known state of agents (sessionId -> info). */
  private lastAgents = new Map<string, RemoteAgentInfo>();
  /** Version string (injected). */
  private readonly version: string;
  /** Hostname. */
  private readonly hostname: string;
  /** Agent info provider callback. */
  private readonly agentInfoProvider: AgentInfoProvider;

  constructor(
    options: {
      version: string;
      hostname?: string;
      agentInfoProvider?: AgentInfoProvider;
    },
  ) {
    this.version = options.version;
    this.hostname = options.hostname ?? "unknown";
    this.agentInfoProvider = options.agentInfoProvider ?? (() => []);
  }

  /** Collect current state of all agents. */
  async collect(): Promise<HeartbeatData> {
    const agents = await this.agentInfoProvider();

    // Update last known state
    this.lastAgents.clear();
    for (const agent of agents) {
      this.lastAgents.set(agent.session_id, agent);
    }

    return {
      agents,
      system_load: 0.0, // Deno doesn't expose CPU load easily
      hostname: this.hostname,
      os: Deno.build.os,
      version: this.version,
    };
  }

  /** Detect changes since last collection. */
  async detectChanges(): Promise<AgentEvent[]> {
    const currentAgentsList = await this.agentInfoProvider();
    const currentAgents = new Map<string, RemoteAgentInfo>();
    for (const a of currentAgentsList) {
      currentAgents.set(a.session_id, a);
    }

    const events: AgentEvent[] = [];

    // Check for new agents (spawned)
    for (const [sessionId, agent] of currentAgents) {
      if (!this.lastAgents.has(sessionId)) {
        events.push({
          event_type: "spawned",
          agent_id: sessionId,
          data: agent,
        });
      }
    }

    // Check for removed agents (exited)
    for (const sessionId of this.lastAgents.keys()) {
      if (!currentAgents.has(sessionId)) {
        events.push({
          event_type: "exited",
          agent_id: sessionId,
          data: {},
        });
      }
    }

    // Check for state changes in existing agents
    for (const [sessionId, current] of currentAgents) {
      const previous = this.lastAgents.get(sessionId);
      if (previous) {
        // Check busy state change
        if (current.is_busy !== previous.is_busy) {
          events.push({
            event_type: current.is_busy ? "busy" : "idle",
            agent_id: sessionId,
            data: { is_busy: current.is_busy, status: current.status },
          });
        }

        // Check for other state changes
        if (
          current.message_count !== previous.message_count ||
          current.status !== previous.status
        ) {
          events.push({
            event_type: "state_changed",
            agent_id: sessionId,
            data: {
              message_count: current.message_count,
              status: current.status,
              previous_message_count: previous.message_count,
              previous_status: previous.status,
            },
          });
        }
      }
    }

    // Update last known state
    this.lastAgents = currentAgents;

    return events;
  }

  /** Get the current list of agents without updating state. */
  getCurrentAgents(): RemoteAgentInfo[] {
    return [...this.lastAgents.values()];
  }

  /** Check if any agents are currently tracked. */
  hasAgents(): boolean {
    return this.lastAgents.size > 0;
  }

  /** Get agent count. */
  agentCount(): number {
    return this.lastAgents.size;
  }
}

// ============================================================================
// Protocol Metrics (telemetry)
// ============================================================================

/** Snapshot of protocol metrics. */
export interface MetricsSnapshot {
  messages_sent: number;
  messages_failed: number;
  bytes_sent: number;
  bytes_received: number;
  compression_ratio: number;
  latency_p50?: number;
  latency_p95?: number;
  latency_p99?: number;
  roundtrip_p50?: number;
  roundtrip_p95?: number;
  roundtrip_p99?: number;
  uptime_secs: number;
  idle_secs: number;
}

/** Connection quality assessment. */
export type ConnectionQuality =
  | "excellent"
  | "good"
  | "fair"
  | "poor"
  | "unknown";

/** Assess connection quality from a metrics snapshot. */
export function assessConnectionQuality(
  snapshot: MetricsSnapshot,
): ConnectionQuality {
  if (snapshot.messages_sent < 10) return "unknown";

  const errorRate = snapshot.messages_sent > 0
    ? snapshot.messages_failed / snapshot.messages_sent
    : 0;
  const latency = snapshot.latency_p95 ?? 0;

  if (errorRate > 0.10 || latency > 250) return "poor";
  if (errorRate > 0.05 || latency > 100) return "fair";
  if (errorRate > 0.01 || latency > 50) return "good";
  return "excellent";
}

const MAX_LATENCY_SAMPLES = 1000;

/** Calculate a percentile from a sorted array. */
function percentile(sorted: number[], p: number): number | undefined {
  if (sorted.length === 0) return undefined;
  const index = Math.round((p / 100) * (sorted.length - 1));
  return sorted[index];
}

/**
 * Protocol metrics for observability.
 * Equivalent to Rust's `ProtocolMetrics`.
 */
export class ProtocolMetrics {
  private latencySamples: number[] = [];
  private roundtripSamples: number[] = [];
  private _messagesSent = 0;
  private _messagesFailed = 0;
  private _bytesSent = 0;
  private _bytesReceived = 0;
  private _bytesUncompressed = 0;
  private _bytesCompressed = 0;
  private connectionStartMs: number | undefined;
  private lastActivityMs: number | undefined;

  /** Record connection start. */
  recordConnectionStart(): void {
    this.connectionStartMs = Date.now();
    this.lastActivityMs = Date.now();
  }

  /** Record message sent. */
  recordMessageSent(bytes: number): void {
    this._messagesSent++;
    this._bytesSent += bytes;
    this.lastActivityMs = Date.now();
  }

  /** Record message failed. */
  recordMessageFailed(): void {
    this._messagesFailed++;
  }

  /** Record bytes received. */
  recordBytesReceived(bytes: number): void {
    this._bytesReceived += bytes;
    this.lastActivityMs = Date.now();
  }

  /** Record compression ratio. */
  recordCompression(uncompressed: number, compressed: number): void {
    this._bytesUncompressed += uncompressed;
    this._bytesCompressed += compressed;
  }

  /** Record message latency (one-way) in milliseconds. */
  recordLatency(latencyMs: number): void {
    if (this.latencySamples.length >= MAX_LATENCY_SAMPLES) {
      this.latencySamples.shift();
    }
    this.latencySamples.push(latencyMs);
  }

  /** Record command roundtrip time in milliseconds. */
  recordRoundtrip(roundtripMs: number): void {
    if (this.roundtripSamples.length >= MAX_LATENCY_SAMPLES) {
      this.roundtripSamples.shift();
    }
    this.roundtripSamples.push(roundtripMs);
  }

  /** Get current metrics snapshot. */
  snapshot(): MetricsSnapshot {
    const now = Date.now();
    const uptimeSecs = this.connectionStartMs !== undefined
      ? Math.floor((now - this.connectionStartMs) / 1000)
      : 0;
    const idleSecs = this.lastActivityMs !== undefined
      ? Math.floor((now - this.lastActivityMs) / 1000)
      : 0;

    const compressionRatio = this._bytesUncompressed > 0
      ? this._bytesCompressed / this._bytesUncompressed
      : 1.0;

    const sortedLatency = [...this.latencySamples].sort((a, b) => a - b);
    const sortedRoundtrip = [...this.roundtripSamples].sort((a, b) => a - b);

    return {
      messages_sent: this._messagesSent,
      messages_failed: this._messagesFailed,
      bytes_sent: this._bytesSent,
      bytes_received: this._bytesReceived,
      compression_ratio: compressionRatio,
      latency_p50: percentile(sortedLatency, 50),
      latency_p95: percentile(sortedLatency, 95),
      latency_p99: percentile(sortedLatency, 99),
      roundtrip_p50: percentile(sortedRoundtrip, 50),
      roundtrip_p95: percentile(sortedRoundtrip, 95),
      roundtrip_p99: percentile(sortedRoundtrip, 99),
      uptime_secs: uptimeSecs,
      idle_secs: idleSecs,
    };
  }

  /** Reset all metrics. */
  reset(): void {
    this.latencySamples = [];
    this.roundtripSamples = [];
    this._messagesSent = 0;
    this._messagesFailed = 0;
    this._bytesSent = 0;
    this._bytesReceived = 0;
    this._bytesUncompressed = 0;
    this._bytesCompressed = 0;
    this.connectionStartMs = undefined;
    this.lastActivityMs = undefined;
  }
}
