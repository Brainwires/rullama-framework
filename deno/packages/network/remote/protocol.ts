/**
 * @module remote/protocol
 *
 * Wire protocol types for CLI <-> Backend communication.
 * Defines the message format for the remote control WebSocket connection.
 *
 * Equivalent to Rust's `rullama-network::remote::protocol`.
 */

// ============================================================================
// Protocol Version Constants
// ============================================================================

/** Current protocol version. */
export const PROTOCOL_VERSION = "1.1";

/** Minimum supported protocol version. */
export const MIN_PROTOCOL_VERSION = "1.0";

/** All supported protocol versions (newest first). */
export const SUPPORTED_VERSIONS: readonly string[] = ["1.1", "1.0"];

// ============================================================================
// Protocol Capabilities
// ============================================================================

/** Capabilities that can be negotiated between CLI and backend. */
export type ProtocolCapability =
  | "streaming"
  | "tools"
  | "presence"
  | "compression"
  | "attachments"
  | "priority"
  | "telemetry";

/** Get all capabilities supported by this client version. */
export function allSupportedCapabilities(): ProtocolCapability[] {
  return ["streaming", "tools", "attachments", "priority"];
}

// ============================================================================
// Command Priority
// ============================================================================

/** Priority level for commands. Lower numeric value = higher priority. */
export type CommandPriority = "critical" | "high" | "normal" | "low";

/** Numeric ordering for priorities (lower = higher priority). */
export const PRIORITY_ORDER: Record<CommandPriority, number> = {
  critical: 0,
  high: 1,
  normal: 2,
  low: 3,
};

/** Retry policy for failed commands. */
export interface RetryPolicy {
  /** Maximum number of retry attempts. */
  max_attempts: number;
  /** Backoff multiplier (e.g., 2.0 for exponential backoff). */
  backoff_multiplier: number;
  /** Initial delay in milliseconds. */
  initial_delay_ms: number;
}

/** Default retry policy. */
export function defaultRetryPolicy(): RetryPolicy {
  return {
    max_attempts: 3,
    backoff_multiplier: 2.0,
    initial_delay_ms: 100,
  };
}

/** Wrapper for prioritized commands. */
export interface PrioritizedCommand {
  /** The underlying command. */
  command: BackendCommand;
  /** Priority level. */
  priority: CommandPriority;
  /** Optional deadline in milliseconds from now. */
  deadline_ms?: number;
  /** Optional retry policy. */
  retry_policy?: RetryPolicy;
}

// ============================================================================
// Protocol Negotiation Messages
// ============================================================================

/** Protocol hello message sent by client during registration. */
export interface ProtocolHello {
  /** Protocol versions supported by this client (newest first). */
  supported_versions: string[];
  /** Preferred protocol version. */
  preferred_version: string;
  /** Capabilities this client supports. */
  capabilities: ProtocolCapability[];
}

/** Create a default ProtocolHello. */
export function defaultProtocolHello(): ProtocolHello {
  return {
    supported_versions: [...SUPPORTED_VERSIONS],
    preferred_version: PROTOCOL_VERSION,
    capabilities: allSupportedCapabilities(),
  };
}

/** Protocol accept message sent by backend in response. */
export interface ProtocolAccept {
  /** Selected protocol version. */
  selected_version: string;
  /** Capabilities enabled for this session. */
  enabled_capabilities: ProtocolCapability[];
}

/** Default ProtocolAccept. */
export function defaultProtocolAccept(): ProtocolAccept {
  return {
    selected_version: PROTOCOL_VERSION,
    enabled_capabilities: ["streaming", "tools"],
  };
}

/** Negotiated protocol state after handshake. */
export class NegotiatedProtocol {
  /** The agreed-upon protocol version. */
  readonly version: string;
  /** Capabilities enabled for this session. */
  readonly capabilities: ProtocolCapability[];

  constructor(version: string, capabilities: ProtocolCapability[]) {
    this.version = version;
    this.capabilities = capabilities;
  }

  /** Check if a capability is enabled. */
  hasCapability(cap: ProtocolCapability): boolean {
    return this.capabilities.includes(cap);
  }

  /** Create from protocol accept response. */
  static fromAccept(accept: ProtocolAccept): NegotiatedProtocol {
    return new NegotiatedProtocol(
      accept.selected_version,
      accept.enabled_capabilities,
    );
  }

  /** Create default negotiated protocol. */
  static default(): NegotiatedProtocol {
    return new NegotiatedProtocol(PROTOCOL_VERSION, ["streaming", "tools"]);
  }
}

// ============================================================================
// CLI -> Backend Messages
// ============================================================================

/** Messages FROM client TO Backend (tagged union via `type` field). */
export type RemoteMessage =
  | RemoteMessage_Register
  | RemoteMessage_Heartbeat
  | RemoteMessage_CommandResult
  | RemoteMessage_AgentEvent
  | RemoteMessage_AgentStream
  | RemoteMessage_Pong
  | RemoteMessage_AttachmentReceived;

export interface RemoteMessage_Register {
  type: "register";
  api_key: string;
  hostname: string;
  os: string;
  version: string;
  protocol?: ProtocolHello;
}

export interface RemoteMessage_Heartbeat {
  type: "heartbeat";
  session_token: string;
  agents: RemoteAgentInfo[];
  system_load: number;
}

export interface RemoteMessage_CommandResult {
  type: "command_result";
  command_id: string;
  success: boolean;
  result?: unknown;
  error?: string;
}

export interface RemoteMessage_AgentEvent {
  type: "agent_event";
  event_type: AgentEventType;
  agent_id: string;
  data: unknown;
}

export interface RemoteMessage_AgentStream {
  type: "agent_stream";
  agent_id: string;
  chunk_type: StreamChunkType;
  content: string;
}

export interface RemoteMessage_Pong {
  type: "pong";
  timestamp: number;
}

export interface RemoteMessage_AttachmentReceived {
  type: "attachment_received";
  attachment_id: string;
  success: boolean;
  file_path?: string;
  error?: string;
}

// ============================================================================
// Backend -> CLI Messages
// ============================================================================

/** Messages FROM Backend TO client (tagged union via `type` field). */
export type BackendCommand =
  | BackendCommand_Authenticated
  | BackendCommand_SendInput
  | BackendCommand_SlashCommand
  | BackendCommand_CancelOperation
  | BackendCommand_Subscribe
  | BackendCommand_Unsubscribe
  | BackendCommand_SpawnAgent
  | BackendCommand_RequestSync
  | BackendCommand_Ping
  | BackendCommand_Disconnect
  | BackendCommand_AuthenticationFailed
  | BackendCommand_AttachmentUpload
  | BackendCommand_AttachmentChunk
  | BackendCommand_AttachmentComplete;

export interface BackendCommand_Authenticated {
  type: "authenticated";
  session_token: string;
  user_id: string;
  refresh_interval_secs: number;
  protocol?: ProtocolAccept;
}

export interface BackendCommand_SendInput {
  type: "send_input";
  command_id: string;
  agent_id: string;
  content: string;
}

export interface BackendCommand_SlashCommand {
  type: "slash_command";
  command_id: string;
  agent_id: string;
  command: string;
  args: string[];
}

export interface BackendCommand_CancelOperation {
  type: "cancel_operation";
  command_id: string;
  agent_id: string;
}

export interface BackendCommand_Subscribe {
  type: "subscribe";
  agent_id: string;
}

export interface BackendCommand_Unsubscribe {
  type: "unsubscribe";
  agent_id: string;
}

export interface BackendCommand_SpawnAgent {
  type: "spawn_agent";
  command_id: string;
  model?: string;
  working_directory?: string;
}

export interface BackendCommand_RequestSync {
  type: "request_sync";
}

export interface BackendCommand_Ping {
  type: "ping";
  timestamp: number;
}

export interface BackendCommand_Disconnect {
  type: "disconnect";
  reason: string;
}

export interface BackendCommand_AuthenticationFailed {
  type: "authentication_failed";
  error: string;
}

export interface BackendCommand_AttachmentUpload {
  type: "attachment_upload";
  command_id: string;
  agent_id: string;
  attachment_id: string;
  filename: string;
  mime_type: string;
  size: number;
  compressed: boolean;
  compression_algorithm?: CompressionAlgorithm;
  chunks_total: number;
}

export interface BackendCommand_AttachmentChunk {
  type: "attachment_chunk";
  attachment_id: string;
  chunk_index: number;
  data: string;
  is_final: boolean;
}

export interface BackendCommand_AttachmentComplete {
  type: "attachment_complete";
  attachment_id: string;
  checksum: string;
}

// ============================================================================
// Shared Types
// ============================================================================

/** Compression algorithms supported for attachments. */
export type CompressionAlgorithm = "zstd" | "gzip";

/** Information about a remote agent. */
export interface RemoteAgentInfo {
  session_id: string;
  model: string;
  is_busy: boolean;
  parent_id?: string;
  working_directory: string;
  message_count: number;
  last_activity: number;
  status: string;
  name?: string;
}

/** Types of agent events. */
export type AgentEventType =
  | "spawned"
  | "exited"
  | "busy"
  | "idle"
  | "state_changed"
  | "viewer_connected"
  | "viewer_disconnected";

/** Types of stream chunks. */
export type StreamChunkType =
  | "text"
  | "thinking"
  | "tool_call"
  | "tool_result"
  | "error"
  | "system"
  | "complete"
  | "history"
  | "user_input";
