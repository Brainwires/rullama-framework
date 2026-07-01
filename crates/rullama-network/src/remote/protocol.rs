//! Protocol types for CLI <-> Backend communication
//!
//! Defines the message format for the remote control WebSocket connection.

use serde::{Deserialize, Serialize};

// ============================================================================
// Protocol Version Constants
// ============================================================================

/// Current protocol version
pub const PROTOCOL_VERSION: &str = "1.1";

/// Minimum supported protocol version
pub const MIN_PROTOCOL_VERSION: &str = "1.0";

/// All supported protocol versions (newest first)
pub const SUPPORTED_VERSIONS: &[&str] = &["1.1", "1.0"];

// ============================================================================
// Protocol Capabilities
// ============================================================================

/// Capabilities that can be negotiated between CLI and backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolCapability {
    /// Real-time streaming of agent output
    Streaming,
    /// Tool execution support
    Tools,
    /// Presence tracking (who's viewing)
    Presence,
    /// Message compression for large payloads
    Compression,
    /// File attachment support
    Attachments,
    /// Command priority queuing
    Priority,
    /// Telemetry and metrics
    Telemetry,
    /// Device allowlist / permission relay
    DeviceAllowlist,
    /// Remote tool approval prompts
    PermissionRelay,
}

impl ProtocolCapability {
    /// Get all capabilities supported by this CLI version
    pub fn all_supported() -> Vec<Self> {
        vec![
            Self::Streaming,
            Self::Tools,
            Self::Attachments,
            Self::Priority,
            Self::DeviceAllowlist,
            Self::PermissionRelay,
        ]
    }
}

// ============================================================================
// Command Priority (Phase 5)
// ============================================================================

/// Priority level for commands
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum CommandPriority {
    /// Critical commands (e.g., emergency stop, security)
    Critical = 0,
    /// High priority (e.g., user-initiated actions)
    High = 1,
    /// Normal priority (default)
    #[default]
    Normal = 2,
    /// Low priority (background tasks)
    Low = 3,
}

/// Retry policy for failed commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Backoff multiplier (e.g., 2.0 for exponential backoff)
    pub backoff_multiplier: f32,
    /// Initial delay in milliseconds
    #[serde(default = "default_initial_delay")]
    pub initial_delay_ms: u64,
}

fn default_initial_delay() -> u64 {
    100
}

fn default_permission_timeout() -> u32 {
    60
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_multiplier: 2.0,
            initial_delay_ms: 100,
        }
    }
}

/// Wrapper for prioritized commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizedCommand {
    /// The underlying command
    pub command: BackendCommand,
    /// Priority level
    #[serde(default)]
    pub priority: CommandPriority,
    /// Optional deadline in milliseconds from now
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
    /// Optional retry policy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_policy: Option<RetryPolicy>,
}

// ============================================================================
// Protocol Negotiation Messages
// ============================================================================

/// Protocol hello message sent by CLI during registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolHello {
    /// Protocol versions supported by this CLI (newest first)
    pub supported_versions: Vec<String>,
    /// Preferred protocol version
    pub preferred_version: String,
    /// Capabilities this CLI supports
    pub capabilities: Vec<ProtocolCapability>,
}

impl Default for ProtocolHello {
    fn default() -> Self {
        Self {
            supported_versions: SUPPORTED_VERSIONS.iter().map(|s| s.to_string()).collect(),
            preferred_version: PROTOCOL_VERSION.to_string(),
            capabilities: ProtocolCapability::all_supported(),
        }
    }
}

/// Protocol accept message sent by backend in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolAccept {
    /// Selected protocol version
    pub selected_version: String,
    /// Capabilities enabled for this session
    pub enabled_capabilities: Vec<ProtocolCapability>,
}

impl Default for ProtocolAccept {
    fn default() -> Self {
        Self {
            selected_version: PROTOCOL_VERSION.to_string(),
            enabled_capabilities: vec![ProtocolCapability::Streaming, ProtocolCapability::Tools],
        }
    }
}

/// Negotiated protocol state after handshake
#[derive(Debug, Clone)]
pub struct NegotiatedProtocol {
    /// The agreed-upon protocol version
    pub version: String,
    /// Capabilities enabled for this session
    pub capabilities: Vec<ProtocolCapability>,
}

impl NegotiatedProtocol {
    /// Check if a capability is enabled
    pub fn has_capability(&self, cap: ProtocolCapability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// Create from protocol accept response
    pub fn from_accept(accept: ProtocolAccept) -> Self {
        Self {
            version: accept.selected_version,
            capabilities: accept.enabled_capabilities,
        }
    }
}

impl Default for NegotiatedProtocol {
    fn default() -> Self {
        Self {
            version: PROTOCOL_VERSION.to_string(),
            capabilities: vec![ProtocolCapability::Streaming, ProtocolCapability::Tools],
        }
    }
}

// ============================================================================
// CLI -> Backend Messages
// ============================================================================

/// Messages FROM CLI TO Backend
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteMessage {
    /// Initial registration with API key and protocol negotiation
    Register {
        /// API key for authentication.
        api_key: String,
        /// Client hostname.
        hostname: String,
        /// Client operating system.
        os: String,
        /// Client version string.
        version: String,
        /// Protocol negotiation (optional for backward compatibility)
        #[serde(skip_serializing_if = "Option::is_none")]
        protocol: Option<ProtocolHello>,
        /// Stable device fingerprint — SHA-256(machine_id || hostname || os).
        /// Used for device allowlist verification by the backend.
        #[serde(skip_serializing_if = "Option::is_none")]
        device_fingerprint: Option<String>,
    },

    /// Regular heartbeat with agent status
    Heartbeat {
        /// Session token for authentication.
        session_token: String,
        /// List of active agents.
        agents: Vec<RemoteAgentInfo>,
        /// Current system load (0.0-1.0).
        system_load: f32,
    },

    /// Response to a command
    CommandResult {
        /// ID of the command being responded to.
        command_id: String,
        /// Whether the command succeeded.
        success: bool,
        /// Result data if successful.
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<serde_json::Value>,
        /// Error message if failed.
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },

    /// Agent event (new agent, agent exit, state change)
    AgentEvent {
        /// Type of agent event.
        event_type: AgentEventType,
        /// ID of the agent this event relates to.
        agent_id: String,
        /// Event-specific data payload.
        data: serde_json::Value,
    },

    /// Stream chunk from agent (for real-time viewing)
    AgentStream {
        /// ID of the agent producing the stream.
        agent_id: String,
        /// Type of stream chunk.
        chunk_type: StreamChunkType,
        /// Chunk content text.
        content: String,
    },

    /// Pong response to backend ping
    Pong {
        /// Timestamp from the original ping.
        timestamp: i64,
    },

    // ========================================================================
    // Permission Relay (Phase 6)
    // ========================================================================
    /// Request permission from the remote user for a dangerous tool call
    PermissionRequest {
        /// Unique request identifier.
        request_id: String,
        /// Agent session ID that wants to execute the tool.
        agent_id: String,
        /// Tool name (e.g., "bash", "delete_file", "git_push").
        tool_name: String,
        /// Human-readable description of what the tool will do.
        action: String,
        /// Detailed context (command, file paths, etc.) as JSON.
        #[serde(default)]
        details: serde_json::Value,
        /// Timeout in seconds (after which the request is auto-denied).
        #[serde(default = "default_permission_timeout")]
        timeout_secs: u32,
    },

    // ========================================================================
    // Attachment Responses (Phase 3)
    // ========================================================================
    /// Acknowledgment that attachment was received
    AttachmentReceived {
        /// The attachment ID
        attachment_id: String,
        /// Whether the attachment was successfully processed
        success: bool,
        /// Path where the file was saved (if successful)
        #[serde(skip_serializing_if = "Option::is_none")]
        file_path: Option<String>,
        /// Error message if failed
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

// ============================================================================
// Backend -> CLI Messages
// ============================================================================

/// Messages FROM Backend TO CLI
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendCommand {
    /// Authenticated - here's your session token and negotiated protocol
    Authenticated {
        /// Session token for subsequent requests.
        session_token: String,
        /// Authenticated user ID.
        user_id: String,
        /// Interval in seconds between heartbeats.
        refresh_interval_secs: u32,
        /// Negotiated protocol (optional for backward compatibility)
        #[serde(skip_serializing_if = "Option::is_none")]
        protocol: Option<ProtocolAccept>,
        /// Device allowlist status returned by backend.
        #[serde(skip_serializing_if = "Option::is_none")]
        device_status: Option<DeviceStatus>,
        /// Organization-level policies (enforced client-side, verified server-side).
        #[serde(skip_serializing_if = "Option::is_none")]
        org_policies: Option<OrgPolicies>,
    },

    /// Send input to an agent
    SendInput {
        /// Unique command identifier.
        command_id: String,
        /// Target agent ID.
        agent_id: String,
        /// Input content to send.
        content: String,
    },

    /// Execute slash command on agent
    SlashCommand {
        /// Unique command identifier.
        command_id: String,
        /// Target agent ID.
        agent_id: String,
        /// Slash command name.
        command: String,
        /// Command arguments.
        args: Vec<String>,
    },

    /// Cancel current operation
    CancelOperation {
        /// Unique command identifier.
        command_id: String,
        /// Target agent ID.
        agent_id: String,
    },

    /// Subscribe to agent stream
    Subscribe {
        /// Agent ID to subscribe to.
        agent_id: String,
    },

    /// Unsubscribe from agent stream
    Unsubscribe {
        /// Agent ID to unsubscribe from.
        agent_id: String,
    },

    /// Spawn new agent
    SpawnAgent {
        /// Unique command identifier.
        command_id: String,
        /// Model to use for the new agent.
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Working directory for the new agent.
        #[serde(skip_serializing_if = "Option::is_none")]
        working_directory: Option<String>,
    },

    /// Request full sync of all agents
    RequestSync,

    /// Ping to check connection health
    Ping {
        /// Timestamp for round-trip measurement.
        timestamp: i64,
    },

    /// Disconnect (server closing)
    Disconnect {
        /// Reason for disconnection.
        reason: String,
    },

    /// Authentication failed
    AuthenticationFailed {
        /// Error message describing the failure.
        error: String,
    },

    // ========================================================================
    // Attachment Commands (Phase 3)
    // ========================================================================
    /// Start of an attachment upload
    AttachmentUpload {
        /// Unique command identifier.
        command_id: String,
        /// Target agent ID.
        agent_id: String,
        /// Unique ID for this attachment
        attachment_id: String,
        /// Original filename
        filename: String,
        /// MIME type (e.g., "text/plain", "image/png")
        mime_type: String,
        /// Total size in bytes (uncompressed)
        size: u64,
        /// Whether the data is compressed
        compressed: bool,
        /// Compression algorithm used (if compressed)
        #[serde(skip_serializing_if = "Option::is_none")]
        compression_algorithm: Option<CompressionAlgorithm>,
        /// Total number of chunks
        chunks_total: u32,
    },

    /// A chunk of attachment data
    AttachmentChunk {
        /// Attachment ID this chunk belongs to
        attachment_id: String,
        /// Chunk index (0-based)
        chunk_index: u32,
        /// Base64-encoded data
        data: String,
        /// Whether this is the final chunk
        is_final: bool,
    },

    // ========================================================================
    // Permission Relay (Phase 6)
    // ========================================================================
    /// Response to a permission request from the remote user
    PermissionResponse {
        /// The request_id from the original PermissionRequest.
        request_id: String,
        /// Whether the tool execution was approved.
        approved: bool,
        /// Remember this decision for the rest of the session.
        #[serde(default)]
        remember_for_session: bool,
        /// If true, always allow this tool (no future prompts).
        #[serde(default)]
        always_allow: bool,
    },

    /// Attachment upload complete - verify checksum
    AttachmentComplete {
        /// ID of the completed attachment.
        attachment_id: String,
        /// SHA-256 checksum of the complete (uncompressed) file
        checksum: String,
    },
}

// ============================================================================
// Device Allowlist & Organization Policies
// ============================================================================

/// Device allowlist status returned by backend during authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceStatus {
    /// Device is recognized and allowed.
    Allowed,
    /// Device is new and pending user approval (approve_new mode).
    PendingApproval,
    /// Device is explicitly blocked.
    Blocked,
}

/// Organization-level policies that override user settings.
///
/// Sent by the backend during authentication. The CLI must enforce these
/// locally for fast UX, but the server also validates server-side.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrgPolicies {
    /// Tools that are always blocked by org policy.
    #[serde(default)]
    pub blocked_tools: Vec<String>,
    /// Whether dangerous tool calls must go through permission relay.
    #[serde(default)]
    pub permission_relay_required: bool,
    /// Device allowlist mode enforced by org (overrides user preference).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_allowlist_mode: Option<String>,
    /// Whether all commands must be audit-logged.
    #[serde(default)]
    pub audit_all_commands: bool,
}

/// Compression algorithms supported for attachments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionAlgorithm {
    /// Zstandard compression (fast, good ratio)
    Zstd,
    /// Gzip compression (widely compatible)
    Gzip,
}

/// Information about a remote agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAgentInfo {
    /// Unique session ID of the agent
    pub session_id: String,
    /// AI model being used (e.g., "claude-3-5-sonnet")
    pub model: String,
    /// Whether the agent is currently processing
    pub is_busy: bool,
    /// Parent agent ID (if this is a child agent)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Working directory of the agent
    pub working_directory: String,
    /// Number of messages in conversation
    pub message_count: usize,
    /// Unix timestamp of last activity
    pub last_activity: i64,
    /// Current status description
    pub status: String,
    /// Agent name (if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Types of agent events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEventType {
    /// New agent spawned
    Spawned,
    /// Agent exited
    Exited,
    /// Agent became busy (processing)
    Busy,
    /// Agent became idle
    Idle,
    /// Agent state changed
    StateChanged,
    /// Agent received viewer connection
    ViewerConnected,
    /// Agent lost viewer connection
    ViewerDisconnected,
}

/// Types of stream chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamChunkType {
    /// Text content from assistant
    Text,
    /// Thinking/reasoning content
    Thinking,
    /// Tool call information
    ToolCall,
    /// Tool result
    ToolResult,
    /// Error message
    Error,
    /// System message
    System,
    /// Stream completed
    Complete,
    /// Initial conversation history (JSON array of messages)
    History,
    /// User input (from TUI or other source)
    UserInput,
}

// ============================================================================
// Device Fingerprint
// ============================================================================

/// Compute a stable device fingerprint: SHA-256(machine_id || hostname || os).
///
/// The fingerprint is stable across reboots but unique per machine.
/// - Linux: reads `/etc/machine-id`
/// - macOS: reads `IOPlatformUUID` via `ioreg`
/// - Windows: reads `MachineGuid` from registry via `reg query`
/// - Fallback: uses hostname + OS (less stable but always available)
pub fn compute_device_fingerprint() -> String {
    use sha2::{Digest, Sha256};

    let machine_id = get_machine_id().unwrap_or_default();
    let hostname = gethostname::gethostname().to_string_lossy().to_string();
    let os = std::env::consts::OS;

    let mut hasher = Sha256::new();
    hasher.update(machine_id.as_bytes());
    hasher.update(hostname.as_bytes());
    hasher.update(os.as_bytes());
    hex::encode(hasher.finalize())
}

/// Read the platform-specific machine identifier.
fn get_machine_id() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/etc/machine-id")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .ok()
            .and_then(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.contains("IOPlatformUUID") {
                        return line
                            .split('=')
                            .nth(1)
                            .map(|s| s.trim().trim_matches('"').to_string());
                    }
                }
                None
            })
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .output()
            .ok()
            .and_then(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.contains("MachineGuid") {
                        return line.split_whitespace().last().map(|s| s.to_string());
                    }
                }
                None
            })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remote_message_serialization() {
        let msg = RemoteMessage::Register {
            api_key: "bw_prod_test123".to_string(),
            hostname: "my-laptop".to_string(),
            os: "linux".to_string(),
            version: "0.7.0".to_string(),
            protocol: None,
            device_fingerprint: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"register\""));
        assert!(json.contains("\"api_key\":\"bw_prod_test123\""));
    }

    #[test]
    fn test_remote_message_with_protocol() {
        let msg = RemoteMessage::Register {
            api_key: "bw_prod_test123".to_string(),
            hostname: "my-laptop".to_string(),
            os: "linux".to_string(),
            version: "0.7.0".to_string(),
            protocol: Some(ProtocolHello::default()),
            device_fingerprint: Some("abc123def456".to_string()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"protocol\""));
        assert!(json.contains("\"preferred_version\":\"1.1\""));
        assert!(json.contains("\"streaming\""));
    }

    #[test]
    fn test_backend_command_deserialization() {
        // Test backward compatibility - no protocol field
        let json = r#"{"type":"authenticated","session_token":"abc123","user_id":"user-456","refresh_interval_secs":30}"#;
        let cmd: BackendCommand = serde_json::from_str(json).unwrap();

        match cmd {
            BackendCommand::Authenticated {
                session_token,
                user_id,
                refresh_interval_secs,
                protocol,
                device_status,
                org_policies,
            } => {
                assert_eq!(session_token, "abc123");
                assert_eq!(user_id, "user-456");
                assert_eq!(refresh_interval_secs, 30);
                assert!(protocol.is_none());
                assert!(device_status.is_none());
                assert!(org_policies.is_none());
            }
            _ => panic!("Expected Authenticated command"),
        }
    }

    #[test]
    fn test_backend_command_with_protocol() {
        let json = r#"{"type":"authenticated","session_token":"abc123","user_id":"user-456","refresh_interval_secs":30,"protocol":{"selected_version":"1.1","enabled_capabilities":["streaming","tools"]}}"#;
        let cmd: BackendCommand = serde_json::from_str(json).unwrap();

        match cmd {
            BackendCommand::Authenticated { protocol, .. } => {
                let proto = protocol.expect("Expected protocol");
                assert_eq!(proto.selected_version, "1.1");
                assert!(
                    proto
                        .enabled_capabilities
                        .contains(&ProtocolCapability::Streaming)
                );
                assert!(
                    proto
                        .enabled_capabilities
                        .contains(&ProtocolCapability::Tools)
                );
            }
            _ => panic!("Expected Authenticated command"),
        }
    }

    #[test]
    fn test_protocol_capability_serialization() {
        let cap = ProtocolCapability::Streaming;
        let json = serde_json::to_string(&cap).unwrap();
        assert_eq!(json, "\"streaming\"");

        let cap: ProtocolCapability = serde_json::from_str("\"attachments\"").unwrap();
        assert_eq!(cap, ProtocolCapability::Attachments);
    }

    #[test]
    fn test_negotiated_protocol() {
        let accept = ProtocolAccept {
            selected_version: "1.1".to_string(),
            enabled_capabilities: vec![
                ProtocolCapability::Streaming,
                ProtocolCapability::Compression,
            ],
        };

        let negotiated = NegotiatedProtocol::from_accept(accept);
        assert!(negotiated.has_capability(ProtocolCapability::Streaming));
        assert!(negotiated.has_capability(ProtocolCapability::Compression));
        assert!(!negotiated.has_capability(ProtocolCapability::Attachments));
    }

    #[test]
    fn test_remote_agent_info() {
        let info = RemoteAgentInfo {
            session_id: "agent-123".to_string(),
            model: "claude-3-5-sonnet".to_string(),
            is_busy: false,
            parent_id: None,
            working_directory: "/home/user/project".to_string(),
            message_count: 5,
            last_activity: 1700000000,
            status: "idle".to_string(),
            name: Some("main-agent".to_string()),
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"session_id\":\"agent-123\""));
        assert!(json.contains("\"name\":\"main-agent\""));
    }
}
