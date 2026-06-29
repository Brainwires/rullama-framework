//! IPC Protocol Definitions
//!
//! Defines the message types for communication between the TUI viewer and Agent process.
//! This is the bridge-crate version, using `rullama_core::ToolMode` instead of
//! CLI-specific types. The `From<ResourceType>` impls live in the CLI adapter.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use rullama_core::ToolMode;

/// Messages sent from Viewer (TUI) to Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ViewerMessage {
    /// User submitted input text
    UserInput {
        /// The user's message text
        content: String,
        /// Files to include in context (from working set)
        #[serde(default)]
        context_files: Vec<String>,
    },

    /// Cancel the current operation (streaming or tool execution)
    Cancel,

    /// Request full conversation state sync
    SyncRequest,

    /// Viewer is detaching (going to background)
    Detach {
        /// If true, agent should exit when current work completes
        exit_when_done: bool,
    },

    /// Request agent to exit immediately
    Exit,

    /// Execute a slash command
    SlashCommand {
        /// Command name (without /)
        command: String,
        /// Command arguments
        args: Vec<String>,
    },

    /// Change tool mode
    SetToolMode {
        /// The new tool mode to set.
        mode: ToolMode,
    },

    /// Queue a message for injection during agent processing
    QueueMessage {
        /// The message content to queue.
        content: String,
    },

    /// Request to acquire a resource lock
    AcquireLock {
        /// Type of resource to lock
        resource_type: ResourceLockType,
        /// Scope of the lock (global or project path)
        scope: String,
        /// Description of the operation
        description: String,
    },

    /// Release a resource lock
    ReleaseLock {
        /// Type of resource to release
        resource_type: ResourceLockType,
        /// Scope of the lock
        scope: String,
    },

    /// Query lock status
    QueryLocks {
        /// Optional scope filter
        scope: Option<String>,
    },

    /// Update lock status message
    UpdateLockStatus {
        /// Type of resource
        resource_type: ResourceLockType,
        /// Scope of the lock
        scope: String,
        /// New status message
        status: String,
    },

    // ========================================================================
    // Multi-Agent Messages
    // ========================================================================
    /// Request list of all active agents
    ListAgents,

    /// Request to spawn a new child agent
    SpawnAgent {
        /// Model for the new agent (defaults to parent's model)
        model: Option<String>,
        /// Reason for spawning (displayed in agent tree)
        reason: Option<String>,
        /// Working directory for the new agent (defaults to parent's)
        working_directory: Option<String>,
    },

    /// Notify child agents on parent exit
    NotifyChildren {
        /// What action children should take
        action: ChildNotifyAction,
    },

    /// Signal from parent to child agent (via IPC or message queue)
    ParentSignal {
        /// The signal type
        signal: ParentSignalType,
        /// Parent's session ID
        parent_session_id: String,
    },

    /// Viewer is disconnecting (graceful close, different from Detach)
    Disconnect,

    // ========================================================================
    // Plan Mode Messages
    // ========================================================================
    /// Enter plan mode with optional focus/goal
    EnterPlanMode {
        /// Optional focus or goal for the planning session
        focus: Option<String>,
    },

    /// Exit plan mode and return to main context
    ExitPlanMode,

    /// User input while in plan mode
    PlanModeUserInput {
        /// The user's message text
        content: String,
        /// Files to include in context
        #[serde(default)]
        context_files: Vec<String>,
    },

    /// Request plan mode state sync
    PlanModeSyncRequest,
}

/// Messages sent from Agent to Viewer (TUI)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentMessage {
    /// Streaming text chunk from AI response
    StreamChunk {
        /// The text delta
        text: String,
    },

    /// Stream completed (full response received)
    StreamEnd {
        /// Optional finish reason
        finish_reason: Option<String>,
    },

    /// Tool call started
    ToolCallStart {
        /// Tool use ID
        id: String,
        /// Tool name
        name: String,
        /// Server name (for MCP tools)
        #[serde(default)]
        server: Option<String>,
        /// Tool input parameters
        input: Value,
    },

    /// Tool execution progress update
    ToolProgress {
        /// Tool name
        name: String,
        /// Progress message
        message: String,
        /// Progress percentage (0.0-1.0) if known
        progress: Option<f64>,
    },

    /// Tool execution completed
    ToolResult {
        /// Tool use ID
        id: String,
        /// Tool name
        name: String,
        /// Result output (if successful)
        output: Option<String>,
        /// Error message (if failed)
        error: Option<String>,
    },

    /// Full conversation state (sent on attach or sync request)
    ConversationSync {
        /// Session ID
        session_id: String,
        /// Current model name
        model: String,
        /// Full conversation history (for display)
        messages: Vec<DisplayMessage>,
        /// Current status message
        status: String,
        /// Whether an operation is in progress
        is_busy: bool,
        /// Current tool mode
        tool_mode: ToolMode,
        /// Connected MCP servers
        mcp_servers: Vec<String>,
    },

    /// New message added to conversation
    MessageAdded {
        /// The message that was added
        message: DisplayMessage,
    },

    /// Status update (e.g., "Working...", "Connected to MCP server X")
    StatusUpdate {
        /// New status message
        status: String,
    },

    /// Task list update
    TaskUpdate {
        /// Formatted task tree for display
        task_tree: String,
        /// Total task count
        task_count: usize,
        /// Completed task count
        completed_count: usize,
    },

    /// Error occurred
    Error {
        /// Error message
        message: String,
        /// Whether this is a fatal error (agent will exit)
        fatal: bool,
    },

    /// Agent is exiting
    Exiting {
        /// Reason for exit
        reason: String,
    },

    /// Acknowledgment of viewer command
    Ack {
        /// Original command type that was acknowledged
        command: String,
    },

    /// Result of a slash command execution (for remote control)
    SlashCommandResult {
        /// The command that was executed (without leading /)
        command: String,
        /// Whether the command executed successfully
        success: bool,
        /// Output/result of the command (for Message or Help results)
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        /// Description of action taken (for Action results)
        #[serde(skip_serializing_if = "Option::is_none")]
        action_taken: Option<String>,
        /// Error message if command failed
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        /// Whether the command was blocked by security policy
        #[serde(default)]
        blocked: bool,
    },

    /// Toast notification
    Toast {
        /// Message to display
        message: String,
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// SEAL status update
    SealStatus {
        /// Whether SEAL is enabled
        enabled: bool,
        /// Entity count
        entity_count: usize,
        /// Last resolution (if any)
        last_resolution: Option<String>,
        /// Quality score (0.0-1.0)
        quality_score: f32,
    },

    /// Lock acquisition result
    LockResult {
        /// Whether the lock was acquired
        success: bool,
        /// Resource type
        resource_type: ResourceLockType,
        /// Scope
        scope: String,
        /// Error message if failed
        error: Option<String>,
        /// Info about blocking lock if failed
        blocking_agent: Option<String>,
    },

    /// Lock released confirmation
    LockReleased {
        /// Resource type
        resource_type: ResourceLockType,
        /// Scope
        scope: String,
    },

    /// Response to QueryLocks
    LockStatus {
        /// All current locks
        locks: Vec<LockInfo>,
    },

    /// Lock state changed notification (for other viewers)
    LockChanged {
        /// The change type
        change: LockChangeType,
        /// Lock info
        lock: LockInfo,
    },

    // ========================================================================
    // Multi-Agent Messages
    // ========================================================================
    /// A new child agent was spawned
    AgentSpawned {
        /// Session ID of the new agent
        new_session_id: String,
        /// Session ID of the parent that spawned it
        parent_session_id: String,
        /// Reason for spawning
        spawn_reason: String,
        /// Model used by the new agent
        model: String,
    },

    /// Response to ListAgents request
    AgentList {
        /// All active agents with their metadata
        agents: Vec<AgentMetadata>,
    },

    /// An agent is exiting (sent to TUI and potentially to children)
    AgentExiting {
        /// Session ID of the exiting agent
        session_id: String,
        /// Reason for exit
        reason: String,
        /// Child agents that were notified
        children_notified: Vec<String>,
    },

    /// Signal received from parent agent
    ParentSignalReceived {
        /// The signal type
        signal: ParentSignalType,
        /// Parent's session ID
        parent_session_id: String,
    },

    // ========================================================================
    // Plan Mode Messages
    // ========================================================================
    /// Plan mode entered successfully
    PlanModeEntered {
        /// Plan session ID
        plan_session_id: String,
        /// Display messages from plan mode
        messages: Vec<DisplayMessage>,
        /// Current status
        status: String,
    },

    /// Plan mode exited
    PlanModeExited {
        /// Optional summary of the planning session
        summary: Option<String>,
    },

    /// Plan mode state sync (response to PlanModeSyncRequest)
    PlanModeSync {
        /// Plan session ID
        plan_session_id: String,
        /// Main session ID
        main_session_id: String,
        /// Display messages from plan mode
        messages: Vec<DisplayMessage>,
        /// Current status
        status: String,
        /// Whether an operation is in progress
        is_busy: bool,
    },

    /// New message added to plan mode conversation
    PlanModeMessageAdded {
        /// The message that was added
        message: DisplayMessage,
    },

    /// Streaming text chunk in plan mode
    PlanModeStreamChunk {
        /// The text delta
        text: String,
    },

    /// Plan mode stream completed
    PlanModeStreamEnd {
        /// Optional finish reason
        finish_reason: Option<String>,
    },
}

/// Type of resource lock (standalone bridge type)
///
/// The CLI provides `From<ResourceType>` impls to convert between this
/// and the CLI's `agents::resource_locks::ResourceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLockType {
    /// Build lock
    Build,
    /// Test lock
    Test,
    /// Combined build+test lock
    BuildTest,
    /// Git index/staging operations
    GitIndex,
    /// Git commit operations
    GitCommit,
    /// Git remote write (push)
    GitRemoteWrite,
    /// Git remote merge (pull)
    GitRemoteMerge,
    /// Git branch operations
    GitBranch,
    /// Git destructive operations
    GitDestructive,
}

impl ResourceLockType {
    /// Convert to string for LockStore
    pub fn as_lock_type_str(&self) -> &'static str {
        match self {
            ResourceLockType::Build => "build",
            ResourceLockType::Test => "test",
            ResourceLockType::BuildTest => "build_test",
            ResourceLockType::GitIndex => "git_index",
            ResourceLockType::GitCommit => "git_commit",
            ResourceLockType::GitRemoteWrite => "git_remote_write",
            ResourceLockType::GitRemoteMerge => "git_remote_merge",
            ResourceLockType::GitBranch => "git_branch",
            ResourceLockType::GitDestructive => "git_destructive",
        }
    }

    /// Parse from string (from LockStore)
    pub fn from_lock_type_str(s: &str) -> Option<Self> {
        match s {
            "build" => Some(ResourceLockType::Build),
            "test" => Some(ResourceLockType::Test),
            "build_test" => Some(ResourceLockType::BuildTest),
            "git_index" => Some(ResourceLockType::GitIndex),
            "git_commit" => Some(ResourceLockType::GitCommit),
            "git_remote_write" => Some(ResourceLockType::GitRemoteWrite),
            "git_remote_merge" => Some(ResourceLockType::GitRemoteMerge),
            "git_branch" => Some(ResourceLockType::GitBranch),
            "git_destructive" => Some(ResourceLockType::GitDestructive),
            _ => None,
        }
    }
}

/// Information about a lock (for IPC)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockInfo {
    /// Agent holding the lock
    pub agent_id: String,
    /// Resource type
    pub resource_type: ResourceLockType,
    /// Scope (global or project path)
    pub scope: String,
    /// Description of operation
    pub description: String,
    /// Current status message
    pub status: String,
    /// Seconds since lock was acquired
    pub held_for_secs: u64,
}

/// Type of lock change notification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LockChangeType {
    /// Lock was acquired
    Acquired,
    /// Lock was released
    Released,
    /// Lock became stale (holder died)
    Stale,
}

/// Display message format (simplified for TUI rendering)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayMessage {
    /// Role of the sender
    pub role: String,
    /// Message content (rendered text)
    pub content: String,
    /// Timestamp (Unix epoch ms)
    pub created_at: i64,
}

impl DisplayMessage {
    /// Create a new display message
    pub fn new(role: impl Into<String>, content: impl Into<String>, created_at: i64) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            created_at,
        }
    }
}

/// Agent configuration sent on startup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Session ID
    pub session_id: String,
    /// Model to use
    pub model: String,
    /// MDAP configuration (if enabled)
    pub mdap_enabled: bool,
    /// SEAL enabled
    pub seal_enabled: bool,
    /// Initial working directory
    pub working_directory: String,
}

/// Handshake message for initial connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handshake {
    /// Protocol version
    pub version: u32,
    /// Whether this is a new session or reattach
    pub is_reattach: bool,
    /// Session ID (for reattach)
    pub session_id: Option<String>,
    /// Session token for authentication (required for reattach)
    /// This is a cryptographically random 64-character hex string
    #[serde(default)]
    pub session_token: Option<String>,
}

impl Handshake {
    /// Current protocol version - bumped to 2 for session token support
    pub const PROTOCOL_VERSION: u32 = 2;

    /// Create a new session handshake
    pub fn new_session() -> Self {
        Self {
            version: Self::PROTOCOL_VERSION,
            is_reattach: false,
            session_id: None,
            session_token: None,
        }
    }

    /// Create a reattach handshake with session token for authentication
    pub fn reattach(session_id: String, session_token: String) -> Self {
        Self {
            version: Self::PROTOCOL_VERSION,
            is_reattach: true,
            session_id: Some(session_id),
            session_token: Some(session_token),
        }
    }
}

/// Handshake response from agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResponse {
    /// Whether the handshake was accepted
    pub accepted: bool,
    /// Session ID (assigned for new sessions, echoed for reattach)
    pub session_id: String,
    /// Session token for authentication (returned to new sessions for later reattachment)
    /// This should be stored securely by the client
    #[serde(default)]
    pub session_token: Option<String>,
    /// Error message if not accepted
    pub error: Option<String>,
}

// ============================================================================
// Multi-Agent Architecture Types
// ============================================================================

/// Metadata about an agent for registry and discovery
///
/// This is stored alongside the agent socket as a `.meta.json` file
/// to enable agent discovery, tree visualization, and lifecycle management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Unique session identifier
    pub session_id: String,
    /// Parent agent that spawned this one (if any)
    pub parent_agent_id: Option<String>,
    /// Reason this agent was spawned (e.g., "Tool call: investigate auth bug")
    pub spawn_reason: Option<String>,
    /// Model being used by this agent
    pub model: String,
    /// Unix timestamp (seconds) when agent was created
    pub created_at: i64,
    /// Unix timestamp (seconds) of last activity
    pub last_activity: i64,
    /// Working directory for this agent
    pub working_directory: String,
    /// Whether the agent is currently busy (processing a request)
    pub is_busy: bool,
    /// Process ID of the agent (for liveness checking)
    #[serde(default)]
    pub pid: Option<u32>,
}

impl AgentMetadata {
    /// Create new metadata for an agent
    pub fn new(session_id: String, model: String, working_directory: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            session_id,
            parent_agent_id: None,
            spawn_reason: None,
            model,
            created_at: now,
            last_activity: now,
            working_directory,
            is_busy: false,
            pid: None,
        }
    }

    /// Set parent agent info
    pub fn with_parent(mut self, parent_id: String, reason: Option<String>) -> Self {
        self.parent_agent_id = Some(parent_id);
        self.spawn_reason = reason;
        self
    }

    /// Set the process ID
    pub fn with_pid(mut self, pid: u32) -> Self {
        self.pid = Some(pid);
        self
    }

    /// Update last activity timestamp
    pub fn touch(&mut self) {
        self.last_activity = chrono::Utc::now().timestamp();
    }

    /// Set busy state
    pub fn set_busy(&mut self, busy: bool) {
        self.is_busy = busy;
        self.touch();
    }
}

/// Action to take when notifying child agents (on parent exit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildNotifyAction {
    /// Shut down if idle, otherwise set exit_when_done
    ShutdownIfIdle,
    /// Force immediate shutdown
    ForceShutdown,
    /// Detach from parent (become orphan, keep running)
    Detach,
}

/// Signal types from parent to child agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParentSignalType {
    /// Parent is exiting, child should decide based on busy state
    ParentExiting,
    /// Parent requests child to shutdown immediately
    Shutdown,
    /// Parent is detaching, child becomes orphan
    Detached,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewer_message_serialization() {
        let msg = ViewerMessage::UserInput {
            content: "Hello".to_string(),
            context_files: vec!["main.rs".to_string()],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user_input"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_agent_message_serialization() {
        let msg = AgentMessage::StreamChunk {
            text: "World".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("stream_chunk"));
        assert!(json.contains("World"));
    }

    #[test]
    fn test_handshake_new_session() {
        let hs = Handshake::new_session();
        assert!(!hs.is_reattach);
        assert!(hs.session_id.is_none());
        assert!(hs.session_token.is_none());
        assert_eq!(hs.version, Handshake::PROTOCOL_VERSION);
    }

    #[test]
    fn test_handshake_reattach() {
        let token = "abc123def456".to_string();
        let hs = Handshake::reattach("session-123".to_string(), token.clone());
        assert!(hs.is_reattach);
        assert_eq!(hs.session_id, Some("session-123".to_string()));
        assert_eq!(hs.session_token, Some(token));
    }

    #[test]
    fn test_viewer_message_cancel() {
        let msg = ViewerMessage::Cancel;
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ViewerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, ViewerMessage::Cancel));
    }

    #[test]
    fn test_agent_message_tool_result() {
        let msg = AgentMessage::ToolResult {
            id: "tool-1".to_string(),
            name: "read_file".to_string(),
            output: Some("content".to_string()),
            error: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: AgentMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            AgentMessage::ToolResult {
                id,
                name,
                output,
                error,
            } => {
                assert_eq!(id, "tool-1");
                assert_eq!(name, "read_file");
                assert_eq!(output, Some("content".to_string()));
                assert!(error.is_none());
            }
            _ => panic!("Expected ToolResult"),
        }
    }
}
