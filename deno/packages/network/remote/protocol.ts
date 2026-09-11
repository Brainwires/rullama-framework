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

  /**
   * Record the outcome of a handshake.
   *
   * @param version Protocol version agreed with the backend.
   * @param capabilities Capabilities the backend enabled for this session.
   */
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

/** Initial registration: authenticates the client and opens protocol negotiation. */
export interface RemoteMessage_Register {
  /** Discriminant. */
  type: "register";
  /** API key used to authenticate this client with the backend. */
  api_key: string;
  /** Hostname of the machine running the client. */
  hostname: string;
  /** Operating system name (e.g. `Deno.build.os`). */
  os: string;
  /** Client version string. */
  version: string;
  /** Versions/capabilities offered for negotiation; omitted by pre-1.1 clients. */
  protocol?: ProtocolHello;
}

/** Periodic liveness report carrying the current agent roster. */
export interface RemoteMessage_Heartbeat {
  /** Discriminant. */
  type: "heartbeat";
  /** Session token issued by {@link BackendCommand_Authenticated}. */
  session_token: string;
  /** Current state of every local agent. */
  agents: RemoteAgentInfo[];
  /** System CPU load, 0.0–1.0 (always 0 in this port — Deno exposes no load average). */
  system_load: number;
}

/** Outcome of a previously received backend command. */
export interface RemoteMessage_CommandResult {
  /** Discriminant. */
  type: "command_result";
  /** `command_id` of the backend command this answers. */
  command_id: string;
  /** Whether the command completed successfully. */
  success: boolean;
  /** Command output when `success` is true. */
  result?: unknown;
  /** Failure detail when `success` is false. */
  error?: string;
}

/** Notification of an agent lifecycle/state change. */
export interface RemoteMessage_AgentEvent {
  /** Discriminant. */
  type: "agent_event";
  /** What happened to the agent. */
  event_type: AgentEventType;
  /** Session id of the affected agent. */
  agent_id: string;
  /** Event-specific payload (e.g. the agent info for `"spawned"`). */
  data: unknown;
}

/** One streamed chunk of an agent's output. */
export interface RemoteMessage_AgentStream {
  /** Discriminant. */
  type: "agent_stream";
  /** Session id of the streaming agent. */
  agent_id: string;
  /** Kind of content in `content`. */
  chunk_type: StreamChunkType;
  /** Chunk text. */
  content: string;
}

/** Reply to a {@link BackendCommand_Ping}. */
export interface RemoteMessage_Pong {
  /** Discriminant. */
  type: "pong";
  /** Timestamp echoed from the ping (ms since epoch). */
  timestamp: number;
}

/** Acknowledges that a chunked attachment upload was fully received. */
export interface RemoteMessage_AttachmentReceived {
  /** Discriminant. */
  type: "attachment_received";
  /** Id of the attachment from {@link BackendCommand_AttachmentUpload}. */
  attachment_id: string;
  /** Whether the attachment was assembled and verified successfully. */
  success: boolean;
  /** Local path where the attachment was written when `success` is true. */
  file_path?: string;
  /** Failure detail when `success` is false. */
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

/** Successful registration: carries the session token and the negotiated protocol. */
export interface BackendCommand_Authenticated {
  /** Discriminant. */
  type: "authenticated";
  /** Token the client must send in subsequent heartbeats. */
  session_token: string;
  /** Backend user the API key belongs to. */
  user_id: string;
  /** How often (seconds) the backend expects a heartbeat. */
  refresh_interval_secs: number;
  /** Version/capabilities the backend selected; absent from pre-1.1 backends. */
  protocol?: ProtocolAccept;
}

/** Deliver user input to a running agent. */
export interface BackendCommand_SendInput {
  /** Discriminant. */
  type: "send_input";
  /** Id to echo back in the {@link RemoteMessage_CommandResult}. */
  command_id: string;
  /** Target agent session id. */
  agent_id: string;
  /** Input text for the agent. */
  content: string;
}

/** Run a slash command (e.g. `/clear`) inside an agent session. */
export interface BackendCommand_SlashCommand {
  /** Discriminant. */
  type: "slash_command";
  /** Id to echo back in the {@link RemoteMessage_CommandResult}. */
  command_id: string;
  /** Target agent session id. */
  agent_id: string;
  /** Command name without the leading slash. */
  command: string;
  /** Positional arguments for the command. */
  args: string[];
}

/** Abort the operation an agent is currently executing. */
export interface BackendCommand_CancelOperation {
  /** Discriminant. */
  type: "cancel_operation";
  /** Id to echo back in the {@link RemoteMessage_CommandResult}. */
  command_id: string;
  /** Target agent session id. */
  agent_id: string;
}

/** Ask the client to forward an agent's events/stream to the backend. */
export interface BackendCommand_Subscribe {
  /** Discriminant. */
  type: "subscribe";
  /** Agent session id to subscribe to. */
  agent_id: string;
}

/** Stop forwarding an agent's events/stream. */
export interface BackendCommand_Unsubscribe {
  /** Discriminant. */
  type: "unsubscribe";
  /** Agent session id to unsubscribe from. */
  agent_id: string;
}

/** Start a new local agent session. */
export interface BackendCommand_SpawnAgent {
  /** Discriminant. */
  type: "spawn_agent";
  /** Id to echo back in the {@link RemoteMessage_CommandResult}. */
  command_id: string;
  /** Model to run the agent with (client default when omitted). */
  model?: string;
  /** Working directory for the agent (client default when omitted). */
  working_directory?: string;
}

/** Ask the client to send a fresh heartbeat with the full agent roster. */
export interface BackendCommand_RequestSync {
  /** Discriminant. */
  type: "request_sync";
}

/** Liveness probe; the client answers with {@link RemoteMessage_Pong}. */
export interface BackendCommand_Ping {
  /** Discriminant. */
  type: "ping";
  /** Send time (ms since epoch), echoed back in the pong. */
  timestamp: number;
}

/** Backend is closing the session; the client should stop and reconnect later. */
export interface BackendCommand_Disconnect {
  /** Discriminant. */
  type: "disconnect";
  /** Why the backend disconnected. */
  reason: string;
}

/** Registration was rejected (bad API key, unsupported protocol, …). */
export interface BackendCommand_AuthenticationFailed {
  /** Discriminant. */
  type: "authentication_failed";
  /** Rejection detail. */
  error: string;
}

/** Announces a chunked file transfer to an agent; followed by `attachment_chunk`s and `attachment_complete`. */
export interface BackendCommand_AttachmentUpload {
  /** Discriminant. */
  type: "attachment_upload";
  /** Id to echo back in the {@link RemoteMessage_CommandResult}. */
  command_id: string;
  /** Agent session the file is for. */
  agent_id: string;
  /** Id shared by every chunk of this transfer. */
  attachment_id: string;
  /** Original file name. */
  filename: string;
  /** MIME type of the file. */
  mime_type: string;
  /** Uncompressed size in bytes. */
  size: number;
  /** Whether chunk data is compressed. */
  compressed: boolean;
  /** Algorithm used when `compressed` is true. */
  compression_algorithm?: CompressionAlgorithm;
  /** Number of chunks that will follow. */
  chunks_total: number;
}

/** One chunk of an in-progress attachment upload. */
export interface BackendCommand_AttachmentChunk {
  /** Discriminant. */
  type: "attachment_chunk";
  /** Transfer this chunk belongs to. */
  attachment_id: string;
  /** Zero-based position of this chunk. */
  chunk_index: number;
  /** Chunk bytes, base64-encoded. */
  data: string;
  /** True on the last chunk of the transfer. */
  is_final: boolean;
}

/** Final message of an attachment upload; the client verifies the checksum and replies with {@link RemoteMessage_AttachmentReceived}. */
export interface BackendCommand_AttachmentComplete {
  /** Discriminant. */
  type: "attachment_complete";
  /** Transfer being finalized. */
  attachment_id: string;
  /** Checksum of the assembled file for verification. */
  checksum: string;
}

// ============================================================================
// Shared Types
// ============================================================================

/** Compression algorithms supported for attachments. */
export type CompressionAlgorithm = "zstd" | "gzip";

/** Information about a remote agent. */
export interface RemoteAgentInfo {
  /** Unique session id of the agent (used as `agent_id` in commands). */
  session_id: string;
  /** Model the agent is running. */
  model: string;
  /** True while the agent is processing a request. */
  is_busy: boolean;
  /** Session id of the parent agent, for spawned sub-agents. */
  parent_id?: string;
  /** Working directory the agent operates in. */
  working_directory: string;
  /** Number of messages in the agent's transcript. */
  message_count: number;
  /** Last activity time (ms since epoch). */
  last_activity: number;
  /** Free-form status string as reported by the host. */
  status: string;
  /** Human-readable agent name. */
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
