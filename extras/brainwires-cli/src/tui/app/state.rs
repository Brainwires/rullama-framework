//! Application State Definitions
//!
//! Contains the core App struct and related types for state management.

use crate::agents::TaskAgentStatus;
use crate::agents::TaskManager;
use crate::commands::CommandExecutor;
use crate::mdap::MdapConfig;
use crate::providers::{Provider, ProviderFactory};
use crate::storage::{TaskStore, VectorDatabase};
use crate::tools::ToolExecutor;
use crate::tui::hotkey_content::HotkeyCategory;
use crate::types::agent::PermissionMode;
use crate::types::message::{Message, MessageContent, Role};
use crate::types::plan_mode::{PlanModeState, SavedMainContext};
use crate::types::question::{QuestionAnswerState, QuestionBlock};
use crate::types::tool::{Tool, ToolMode};
use crate::utils::checkpoint::CheckpointManager;
use crate::utils::prompt_history::PromptHistory;
use crate::utils::system_prompt::build_system_prompt;
use anyhow::{Context, Result};
use ratatui_interact::components::ScrollableContentState;
use ratatui_interact::components::hotkey_dialog::HotkeyDialogState;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Simple message for TUI display
#[derive(Debug, Clone)]
pub struct TuiMessage {
    pub role: String,
    pub content: String,
    pub created_at: i64,
}

/// Shell command execution record
#[derive(Debug, Clone)]
pub struct ShellExecution {
    /// The command that was executed
    pub command: String,
    /// Command output (stdout + stderr)
    pub output: String,
    /// Exit code
    pub exit_code: i32,
    /// When the command was executed
    pub executed_at: i64,
}

/// Tool execution entry for Journal display
#[derive(Debug, Clone)]
pub struct ToolExecutionEntry {
    /// Name of the tool that was executed
    pub tool_name: String,
    /// Brief summary of the parameters (for display)
    pub parameters_summary: String,
    /// Brief summary of the result (truncated for display)
    pub result_summary: String,
    /// Whether the tool execution succeeded
    pub success: bool,
    /// When the tool was executed
    pub executed_at: i64,
    /// Duration in milliseconds (if available)
    pub duration_ms: Option<u64>,
}

// ── Sub-Agent Viewer State ────────────────────────────────────────────────────

/// Which panel is focused in the Sub-Agent Viewer
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SubAgentPanelFocus {
    #[default]
    Left,
    Right,
}

/// Display-ready summary of one sub-agent
#[derive(Debug, Clone)]
pub struct SubAgentSummary {
    /// Agent identifier
    pub agent_id: String,
    /// Task description (truncated to 60 chars for display)
    pub task_desc: String,
    /// Current status
    pub status: TaskAgentStatus,
    /// Number of AI iterations executed
    pub iterations: u32,
    /// Session ID if this agent is session-backed
    pub session_id: Option<String>,
    /// Whether the agent's IPC socket is accessible
    pub has_ipc_socket: bool,
}

/// State for the Sub-Agent Viewer mode
#[derive(Debug, Clone, Default)]
pub struct SubAgentViewerState {
    /// Agents list (refreshed on entry and periodically)
    pub agent_list: Vec<SubAgentSummary>,
    /// Selected index in the left panel list
    pub selected_index: usize,
    /// Scroll offset for the left panel
    pub list_scroll: u16,
    /// Scroll offset for the right panel (agent detail)
    pub detail_scroll: u16,
    /// Input content for sending a message to an interactive agent
    pub message_input: String,
    /// Which panel has focus
    pub panel_focus: SubAgentPanelFocus,
}

/// Severity level for activity journal entries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn label(self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO ",
            LogLevel::Warn => "WARN ",
            LogLevel::Error => "ERROR",
        }
    }
}

/// Application state
pub struct App {
    /// Current session ID
    pub session_id: String,
    /// Conversation messages
    pub messages: Vec<TuiMessage>,
    /// Full message history for API calls
    pub(super) conversation_history: Vec<Message>,
    /// Input state (text, cursor, scroll) — backed by ratatui-interact TextAreaState
    pub input_state: ratatui_interact::components::TextAreaState,
    /// Draft input preserved while navigating history (restored when returning to "new" entry)
    pub(super) input_draft: Option<String>,
    /// Scroll position for conversation
    pub scroll: u16,
    /// Active tool executions
    pub active_tools: Vec<ToolExecution>,
    /// Application mode
    pub mode: AppMode,
    /// Prompt history manager
    pub(super) prompt_history: PromptHistory,
    /// Search query (for Ctrl+R mode)
    pub search_query: String,
    /// Search results (for Ctrl+R mode)
    pub(super) search_results: Vec<String>,
    /// Current search result index
    pub(super) search_result_index: usize,
    /// Status message
    pub status: String,
    /// AI Provider
    pub(super) provider: Arc<dyn Provider>,
    /// Model to use
    pub model: String,
    /// Whether to exit
    pub(super) should_quit: bool,
    /// Command executor
    pub(super) command_executor: CommandExecutor,
    /// Checkpoint manager
    pub(super) checkpoint_manager: CheckpointManager,
    /// Conversation store
    pub(super) conversation_store: crate::storage::ConversationStore,
    /// Message store
    pub(super) message_store: crate::storage::MessageStore,
    /// Cleared conversation history (for /resume)
    pub(super) cleared_messages: Option<Vec<TuiMessage>>,
    /// Cleared conversation history (for /resume)
    pub(super) cleared_conversation_history: Option<Vec<Message>>,
    /// Available conversations (for session picker)
    pub available_sessions: Vec<crate::storage::ConversationMetadata>,
    /// Selected session index in picker
    pub selected_session_index: usize,
    /// Session picker scroll position
    pub session_picker_scroll: u16,
    /// Console state (debug output, logs, scrolling)
    pub console_state: ScrollableContentState,
    /// Autocomplete suggestions for slash commands or arguments
    pub autocomplete_suggestions: Vec<String>,
    /// Selected autocomplete index
    pub autocomplete_index: usize,
    /// Whether autocomplete popup is visible
    pub show_autocomplete: bool,
    /// Title for autocomplete popup (e.g., "Slash Commands" or "Models")
    pub autocomplete_title: String,
    /// Cached model IDs for autocomplete
    pub cached_model_ids: Vec<String>,
    /// Active plan metadata (for /plan mode tracking)
    pub active_plan: Option<crate::types::plan::PlanMetadata>,
    /// Completed plan steps (indices)
    pub completed_plan_steps: Vec<usize>,
    /// Approval mode for AI actions
    pub approval_mode: ApprovalMode,
    /// Pending exec command to run in overlay
    pub pending_exec_command: Option<String>,
    /// Shell command execution history
    pub shell_history: Vec<ShellExecution>,
    /// Tool execution history for Journal display
    pub tool_execution_history: Vec<ToolExecutionEntry>,
    /// Selected shell history index (for viewer)
    pub selected_shell_index: usize,
    /// Shell viewer scroll position
    pub shell_viewer_scroll: u16,
    /// Currently focused panel
    pub focused_panel: FocusedPanel,
    /// Task manager for tracking tasks
    pub task_manager: Arc<RwLock<TaskManager>>,
    /// Task store for persistence
    pub(super) task_store: TaskStore,
    /// Cached task tree for UI display (updated when tasks change)
    pub task_tree_cache: String,
    /// Number of tasks in cache
    pub task_count_cache: usize,
    /// Session-specific task list (in-memory only, cleared on session end)
    pub session_tasks: Arc<RwLock<crate::types::session_task::SessionTaskList>>,
    /// Cached summary for status bar display
    pub session_task_summary: String,
    /// Cached panel content for sidebar display: (icon, text, status)
    pub session_task_panel_cache: Vec<(
        String,
        String,
        crate::types::session_task::SessionTaskStatus,
    )>,
    /// Queued user messages (for injecting during agent processing)
    pub(super) queued_messages: Vec<String>,
    /// Plan progress (completed, total)
    pub plan_progress: Option<(usize, usize)>,
    /// Available tools for AI
    pub(super) tools: Vec<Tool>,
    /// Tool executor for running tools (Arc for background task sharing)
    pub(super) tool_executor: Arc<ToolExecutor>,
    /// Current working directory
    pub working_directory: String,
    /// Channel receiver for streaming events from background task
    pub stream_rx: Option<mpsc::UnboundedReceiver<StreamEvent>>,
    /// Accumulated response content during streaming
    pub streaming_content: String,
    /// Index of assistant message being streamed to
    pub streaming_msg_idx: Option<usize>,
    /// Conversation clone for tool handling (saved when stream starts)
    pub(super) streaming_conversation: Option<Vec<Message>>,
    /// User content for saving to storage after stream completes
    pub(super) streaming_user_content: Option<String>,
    /// Toast notification message (shown temporarily)
    pub toast_message: Option<String>,
    /// Toast expiration time (epoch ms)
    pub toast_expires_at: Option<i64>,
    /// Whether mouse capture is disabled (allows terminal text selection)
    pub mouse_capture_disabled: bool,
    /// Working set of files currently in context
    pub working_set: crate::types::WorkingSet,
    /// Last known conversation panel area (for mouse hit testing)
    pub conversation_area: Option<ratatui::layout::Rect>,
    /// Last known input panel area (for mouse hit testing)
    pub input_area: Option<ratatui::layout::Rect>,
    /// Last known status bar area (for mouse hit testing)
    pub status_bar_area: Option<ratatui::layout::Rect>,
    /// Last known exit button area within status bar (for mouse hit testing)
    pub exit_button_area: Option<ratatui::layout::Rect>,
    /// Whether the status bar is currently visible (hidden on small terminals)
    pub status_bar_visible: bool,
    /// Last mouse click time for double-click detection (epoch ms)
    pub last_click_time: Option<i64>,
    /// Last mouse click position for double-click detection (col, row)
    pub last_click_pos: Option<(u16, u16)>,
    /// Cached line count for scroll calculation (updated during render)
    pub conversation_line_count: usize,
    /// Flag to scroll to bottom on next render (set when loading a session)
    pub pending_scroll_to_bottom: bool,
    /// Flag to scroll to bottom on next resize (for PTY mode where initial render has wrong size)
    pub scroll_to_bottom_on_resize: bool,
    /// Flag indicating we're running in PTY mode (backgrounded session)
    pub is_pty_session: bool,
    /// Current prompt mode (Ask / Edit / Plan)
    pub prompt_mode: PromptMode,
    /// Saved prompt mode before entering PlanMode (for restore on exit)
    pub(super) pre_plan_prompt_mode: Option<PromptMode>,
    /// Current tool selection mode
    pub tool_mode: ToolMode,
    /// Tool picker state (for explicit mode UI)
    pub tool_picker_state: Option<ToolPickerState>,
    /// MCP tools from connected servers
    pub mcp_tools: Vec<Tool>,
    /// Connected MCP server names
    pub mcp_connected_servers: Vec<String>,
    /// Channel receiver for tool execution events from background task
    pub tool_rx: Option<mpsc::UnboundedReceiver<StreamEvent>>,
    /// Channel sender for tool execution events (kept alive during tool execution)
    pub(super) tool_tx: Option<mpsc::UnboundedSender<StreamEvent>>,
    /// Pending tool call data for continuation after tool completes
    pub(super) pending_tool_data: Option<PendingToolData>,
    /// Cancellation token for current operation (streaming or tool execution)
    pub(super) cancellation_token: Option<CancellationToken>,
    /// JoinHandle for tool execution task (for abort on cancellation)
    pub(super) tool_task_handle: Option<JoinHandle<()>>,
    /// JoinHandle for streaming task (for abort on cancellation)
    pub(super) stream_task_handle: Option<JoinHandle<()>>,
    /// Task viewer state
    pub task_viewer_state: TaskViewerState,
    /// SEAL status information (displayed when show_status is enabled)
    pub seal_status: SealStatus,
    /// File explorer state
    pub file_explorer_state: Option<super::file_explorer::FileExplorerState>,
    /// Nano editor state
    pub nano_editor_state: Option<super::nano_editor::NanoEditorState>,
    /// Git SCM state
    pub git_scm_state: Option<super::git_scm::GitScmState>,
    /// Find/Replace dialog state
    pub find_replace_state: Option<super::find_replace::FindReplaceState>,
    /// MDAP configuration for high-reliability mode
    pub mdap_config: Option<MdapConfig>,
    /// Pending clarifying questions from AI response
    pub pending_questions: Option<QuestionBlock>,
    /// State for answering clarifying questions
    pub question_state: QuestionAnswerState,
    /// Conversation view style (journal or classic)
    pub conversation_view_style: ConversationViewStyle,
    /// Help dialog state
    pub help_dialog_state: Option<super::help_dialog::HelpDialogState>,
    /// Suspend dialog state
    pub suspend_dialog_state: Option<super::suspend_dialog::SuspendDialogState>,
    /// Exit dialog state
    pub exit_dialog_state: Option<super::exit_dialog::ExitDialogState>,
    /// Hotkey dialog state
    pub hotkey_dialog_state: Option<HotkeyDialogState<HotkeyCategory>>,
    /// Approval dialog state
    pub approval_dialog_state: Option<super::approval_dialog::ApprovalDialogState>,
    /// Channel receiver for approval requests from tool executor
    pub approval_rx: Option<mpsc::Receiver<crate::approval::ApprovalRequest>>,
    /// Sudo password dialog state
    pub sudo_dialog_state: Option<super::sudo_dialog::SudoDialogState>,
    /// Channel receiver for sudo password requests from tool executor
    pub sudo_password_rx: Option<mpsc::Receiver<crate::sudo::SudoPasswordRequest>>,
    /// Channel receiver for `ask_user_question` tool calls.
    pub user_question_rx: Option<mpsc::Receiver<crate::ask::UserQuestionRequest>>,
    /// In-flight user question (if any). Holds the synthetic QuestionBlock
    /// used by the existing question panel renderer, plus the oneshot
    /// response channel we must reply on when the user submits.
    pub active_user_question: Option<super::user_question::PendingUserQuestion>,
    /// Optional shell command whose stdout is appended to the status bar.
    /// Sourced from `Config.status_line_command`; refreshed asynchronously.
    pub status_line_command: Option<String>,
    /// (refreshed_at, text) cache for the custom status line. Avoids
    /// spawning a process on every frame.
    pub status_line_cache: Option<(std::time::Instant, String)>,
    /// Set by the `/shell` slash command — the main TUI loop drops into
    /// an interactive shell and clears this flag on return.
    pub pending_shell: bool,
    /// User-configurable keybinding map. Loaded from layered `settings.json`
    /// at App construction; falls back to defaults if unset.
    pub keybindings: std::sync::Arc<crate::tui::keybindings::KeybindingMap>,
    /// Flag to signal main loop to background the process
    pub pending_background: bool,
    /// Flag to signal main loop to suspend the process
    pub pending_suspend: bool,
    /// If true, exit the agent when it completes (used with backgrounding)
    pub exit_when_agent_done: bool,
    /// If true, preserve chat output on exit instead of restoring previous terminal
    pub preserve_chat_on_exit: bool,
    /// Last known preserve_chat setting (persisted across dialog opens)
    pub last_preserve_chat_setting: bool,
    /// Flag to signal main loop to resume AI response (after reattach with pending user message)
    pub pending_resume_ai: bool,
    /// SEAL processor for query enhancement
    pub(super) seal_processor: Option<brainwires_seal::SealProcessor>,
    /// Dialog state for SEAL coreference resolution
    pub(super) seal_dialog_state: brainwires_seal::DialogState,
    /// Entity store for SEAL
    pub(super) seal_entity_store: crate::utils::entity_extraction::EntityStore,
    /// Entity extractor for SEAL
    pub(super) seal_entity_extractor: crate::utils::entity_extraction::EntityExtractor,

    // Multi-Agent System Fields
    /// Pending agent switch request (session_id)
    pub pending_agent_switch: Option<String>,
    /// Pending agent spawn request (model, reason)
    pub pending_agent_spawn: Option<(Option<String>, Option<String>)>,
    /// IPC writer to current agent (if any) - reader is handled by EventHandler
    pub ipc_writer: Option<brainwires::agent_network::ipc::IpcWriter>,
    /// Whether we're in IPC mode (connected to a separate agent process)
    pub is_ipc_mode: bool,
    /// Flag indicating Session connection was lost and needs respawn
    pub ipc_needs_respawn: bool,
    /// Skill registry for agent skills
    pub skill_registry: Option<brainwires_skills::SkillRegistry>,
    /// Tool allowlist scoped to the next AI turn, set by `/skill` when the
    /// invoked skill declared `allowed_tools`. Cleared after the next
    /// successful AI response so subsequent turns see the full tool set again.
    pub pending_skill_tool_scope: Option<Vec<String>>,

    // Journal Tree Fields
    /// Collapsible tree for the Journal view
    pub journal_tree: super::journal_tree::JournalTreeState,
    /// Sub-Agent Viewer state (Some when mode == AppMode::SubAgentViewer)
    pub sub_agent_viewer_state: Option<SubAgentViewerState>,

    // Plan Mode Fields
    /// Plan mode state (when active)
    pub plan_mode_state: Option<PlanModeState>,
    /// Saved main context when entering plan mode (for restoration)
    pub(super) plan_mode_saved_main: Option<SavedMainContext>,
    /// PKS integration for implicit fact detection and behavioral inference
    pub pks_integration: brainwires::knowledge::bks_pks::personal::PksIntegration,
    /// Count of unread Warn/Error journal entries (cleared when journal is opened)
    pub unread_error_count: usize,
}

/// SEAL processing status for TUI display
#[derive(Debug, Clone, Default)]
pub struct SealStatus {
    /// Whether SEAL is enabled
    pub enabled: bool,
    /// Last coreference resolution (e.g., "it" → "main.rs")
    pub last_resolution: Option<String>,
    /// Number of entities tracked in current conversation
    pub entity_count: usize,
    /// Last matched pattern (from learning)
    pub matched_pattern: Option<String>,
    /// Current quality score (0.0-1.0)
    pub quality_score: f32,
    /// Whether to show SEAL status in UI
    pub show_status: bool,
}

/// State for the task viewer modal
#[derive(Debug, Clone, Default)]
pub struct TaskViewerState {
    /// Flattened task list for navigation (task_id, depth, is_last_sibling)
    pub visible_tasks: Vec<(String, usize, bool)>,
    /// Currently selected task index
    pub selected_index: usize,
    /// Scroll offset
    pub scroll: u16,
    /// Collapsed task IDs (children hidden)
    pub collapsed: std::collections::HashSet<String>,
}

/// Approval mode for AI actions
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalMode {
    /// Review and approve all AI actions (safest)
    Suggest,
    /// Auto-approve file edits, review other actions
    AutoEdit,
    /// Auto-approve all actions (least safe)
    FullAuto,
}

/// Which panel is currently focused
#[derive(Debug, Clone, PartialEq)]
pub enum FocusedPanel {
    /// Conversation panel (Up/Down scrolls)
    Conversation,
    /// Input panel (Up/Down navigates history)
    Input,
    /// Status bar (Exit button focused)
    StatusBar,
}

/// Conversation view style
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ConversationViewStyle {
    /// Journal style - assistant is default speaker, minimal badges
    #[default]
    Journal,
    /// Classic style - all messages have role badges
    Classic,
}

/// Prompt mode — controls how user input is handled (orthogonal to AppMode)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum PromptMode {
    /// Ask mode: read-only, explanation/analysis only, no file mutations
    Ask,
    /// Edit mode: full tool access (default behavior)
    #[default]
    Edit,
    /// Plan mode: isolated planning context
    Plan,
}

/// Application mode
#[derive(Debug, Clone, PartialEq)]
pub enum AppMode {
    /// Normal input mode
    Normal,
    /// Reverse search mode (Ctrl+R)
    ReverseSearch,
    /// Session picker mode
    SessionPicker,
    /// Console view mode (Ctrl+D)
    ConsoleView,
    /// Shell history viewer mode
    ShellViewer,
    /// Waiting for AI response
    Waiting,
    /// Full-screen conversation view (Ctrl+Enter when conversation focused)
    ConversationFullscreen,
    /// Full-screen input view (Ctrl+Enter when input focused)
    InputFullscreen,
    /// Tool picker mode (/tools explicit)
    ToolPicker,
    /// Task viewer modal (Ctrl+T)
    TaskViewer,
    /// File explorer mode (Ctrl+F in fullscreen)
    FileExplorer,
    /// Nano-style file editor mode
    NanoEditor,
    /// Git SCM mode (Ctrl+G in fullscreen)
    GitScm,
    /// Cancel confirmation mode (shown when Esc pressed during Waiting)
    CancelConfirm,
    /// Answering clarifying questions from AI
    QuestionAnswer,
    /// Answering a one-shot `ask_user_question` tool call. Uses the same
    /// panel renderer as `QuestionAnswer` but submission routes the answer
    /// back over a oneshot channel instead of continuing the AI turn.
    UserQuestion,
    /// Find dialog mode (Ctrl+F in fullscreen modes)
    FindDialog,
    /// Find and Replace dialog mode (Ctrl+H in InputFullscreen only)
    FindReplaceDialog,
    /// Help dialog mode (F1 or ?)
    HelpDialog,
    /// Suspend/Background dialog mode (Ctrl+Z)
    SuspendDialog,
    /// Exit/Background dialog mode (Ctrl+C)
    ExitDialog,
    /// Hotkey configuration dialog mode (/hotkeys)
    HotkeyDialog,
    /// Tool approval dialog mode
    ApprovalDialog,
    /// Sudo password dialog mode
    SudoPasswordDialog,
    /// Plan mode - isolated planning context with separate conversation
    PlanMode,
    /// Sub-Agent Viewer (Ctrl+B) - view and interact with sub-agents
    SubAgentViewer,
}

/// A single tool entry: (tool_name, description, selected)
pub type ToolEntry = (String, String, bool);
/// A category with its tools: (category_name, tools)
pub type ToolCategory = (String, Vec<ToolEntry>);

/// State for the tool picker UI (explicit mode)
#[derive(Debug, Clone, Default)]
pub struct ToolPickerState {
    /// Categories with their tools: (category_name, [(tool_name, description, selected)])
    pub categories: Vec<ToolCategory>,
    /// Current selection index (category_index, tool_index within category, or None if on category header)
    pub selected_category: usize,
    /// Selected tool index within category (None means category header is selected)
    pub selected_tool: Option<usize>,
    /// Scroll position
    pub scroll: u16,
    /// Filter/search query
    pub filter_query: String,
    /// Collapsed categories
    pub collapsed: std::collections::HashSet<usize>,
}

/// Pending tool call data stored while tool executes in background
#[derive(Debug, Clone)]
pub struct PendingToolData {
    pub call_id: String,
    pub response_id: String,
    pub chat_id: Option<String>,
    pub tool_name: String,
    pub parameters: serde_json::Value,
    pub conversation: Vec<Message>,
}

/// Tool execution status
#[derive(Debug, Clone)]
pub struct ToolExecution {
    pub tool_name: String,
    pub status: ToolStatus,
    pub result: Option<String>,
}

/// Tool execution status
#[derive(Debug, Clone, PartialEq)]
pub enum ToolStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Events sent from background streaming task to main loop
#[derive(Debug)]
pub enum StreamEvent {
    /// Text chunk received
    Text(String),
    /// Tool call received
    ToolCall {
        call_id: String,
        response_id: String,
        chat_id: Option<String>,
        tool_name: String,
        server: String,
        parameters: serde_json::Value,
    },
    /// Tool execution completed in background
    ToolResult {
        tool_name: String,
        result: Option<String>,
        error: Option<String>,
    },
    /// Progress update from tool execution (MCP progress notifications)
    Progress {
        tool_name: String,
        message: String,
        progress: Option<f64>, // 0.0 to 1.0, if known
    },
    /// Stream completed
    Done,
    /// Error occurred
    Error(String),
}

impl App {
    /// Create a new TUI app in "viewer" mode
    ///
    /// This creates a lightweight App that connects to a Session (Agent) via IPC.
    /// The Session handles all AI/tools/MCP - the TUI is just a viewer.
    ///
    /// Key differences from `new()`:
    /// - No AI provider (provider field is a dummy)
    /// - No tool executor (tools are executed by Session)
    /// - No MCP connections (managed by Session)
    /// - Communicates with Session via IPC
    pub async fn new_viewer(session_id: String, model: Option<String>) -> Result<Self> {
        // Load config for defaults
        let config_manager = crate::config::ConfigManager::new()?;
        let config = config_manager.get();

        // Use model from config if not provided
        let model = model.unwrap_or_else(|| config.model.clone());

        // Load SEAL settings
        let seal_settings = &config.seal;

        // Initialize prompt history
        let prompt_history = PromptHistory::new()?;

        // Initialize storage (for session management, not AI state)
        let db_path = crate::utils::paths::PlatformPaths::conversations_db_path()?;
        let client = std::sync::Arc::new(
            crate::storage::LanceDatabase::new(db_path.to_str().context("Invalid DB path")?)
                .await
                .context("Failed to create LanceDB client")?,
        );
        client
            .initialize(384)
            .await
            .context("Failed to initialize LanceDB")?;
        let embeddings = std::sync::Arc::new(
            crate::storage::embeddings::CachedEmbeddingProvider::new()
                .context("Failed to create embedding provider")?,
        );

        let conversation_store = crate::storage::ConversationStore::new(client.clone());
        let message_store = crate::storage::MessageStore::new(client.clone(), embeddings);
        let task_store = TaskStore::new(client.clone());
        let task_manager = Arc::new(RwLock::new(TaskManager::new()));

        // Create a dummy provider - viewer mode doesn't use it
        // All AI calls go through IPC to the Session
        let provider_factory = ProviderFactory::new();
        let provider = provider_factory.create(model.clone()).await?;

        // Create minimal tool executor - viewer mode doesn't use it
        let tool_executor = Arc::new(ToolExecutor::new(PermissionMode::Auto));
        let working_directory = std::env::current_dir()?.to_string_lossy().to_string();

        // Session task list (not wired to executor in viewer mode)
        let session_tasks = Arc::new(RwLock::new(
            crate::types::session_task::SessionTaskList::new(),
        ));

        // Command executor for local commands
        let command_executor = CommandExecutor::new()?;
        let checkpoint_manager = CheckpointManager::new()?;

        let mut result = Ok(Self {
            session_id,
            messages: Vec::new(),
            conversation_history: Vec::new(), // Managed by Session
            input_state: ratatui_interact::components::TextAreaState::empty(),
            input_draft: None,
            scroll: 0,
            active_tools: Vec::new(),
            mode: AppMode::Normal,
            prompt_history,
            search_query: String::new(),
            search_results: Vec::new(),
            search_result_index: 0,
            status: format!("Connecting to session... Model: {}", model),
            provider,
            model,
            should_quit: false,
            command_executor,
            checkpoint_manager,
            conversation_store,
            message_store,
            cleared_messages: None,
            cleared_conversation_history: None,
            available_sessions: Vec::new(),
            selected_session_index: 0,
            session_picker_scroll: 0,
            console_state: ScrollableContentState::new(vec![
                "TUI running in viewer mode - connected to Session".to_string(),
            ]),
            autocomplete_suggestions: Vec::new(),
            autocomplete_index: 0,
            show_autocomplete: false,
            autocomplete_title: "Slash Commands".to_string(),
            cached_model_ids: Vec::new(),
            active_plan: None,
            completed_plan_steps: Vec::new(),
            approval_mode: ApprovalMode::Suggest,
            pending_exec_command: None,
            shell_history: Vec::new(),
            tool_execution_history: Vec::new(),
            selected_shell_index: 0,
            shell_viewer_scroll: 0,
            focused_panel: FocusedPanel::Input,
            task_manager,
            task_store,
            task_tree_cache: "No tasks".to_string(),
            task_count_cache: 0,
            session_tasks,
            session_task_summary: String::new(),
            session_task_panel_cache: Vec::new(),
            queued_messages: Vec::new(),
            plan_progress: None,
            tools: Vec::new(), // Managed by Session
            tool_executor,
            working_directory,
            stream_rx: None,
            streaming_content: String::new(),
            streaming_msg_idx: None,
            streaming_conversation: None,
            streaming_user_content: None,
            toast_message: None,
            toast_expires_at: None,
            mouse_capture_disabled: false,
            working_set: crate::types::WorkingSet::new(),
            conversation_area: None,
            input_area: None,
            status_bar_area: None,
            exit_button_area: None,
            status_bar_visible: true,
            last_click_time: None,
            last_click_pos: None,
            conversation_line_count: 0,
            pending_scroll_to_bottom: false,
            scroll_to_bottom_on_resize: false,
            is_pty_session: false,
            prompt_mode: PromptMode::default(),
            pre_plan_prompt_mode: None,
            tool_mode: ToolMode::Smart,
            tool_picker_state: None,
            mcp_tools: Vec::new(),             // Managed by Session
            mcp_connected_servers: Vec::new(), // Managed by Session
            tool_rx: None,
            tool_tx: None,
            pending_tool_data: None,
            cancellation_token: None,
            tool_task_handle: None,
            stream_task_handle: None,
            task_viewer_state: TaskViewerState::default(),
            seal_status: SealStatus {
                enabled: seal_settings.enabled,
                last_resolution: None,
                entity_count: 0,
                matched_pattern: None,
                quality_score: 1.0,
                show_status: seal_settings.show_status,
            },
            file_explorer_state: None,
            nano_editor_state: None,
            git_scm_state: None,
            find_replace_state: None,
            mdap_config: None, // Managed by Session
            pending_questions: None,
            question_state: QuestionAnswerState::default(),
            conversation_view_style: ConversationViewStyle::default(),
            help_dialog_state: None,
            suspend_dialog_state: None,
            exit_dialog_state: None,
            hotkey_dialog_state: None,
            approval_dialog_state: None,
            approval_rx: None, // Approval handled by Session
            sudo_dialog_state: None,
            sudo_password_rx: None, // Sudo handled by Session
            user_question_rx: None,
            active_user_question: None,
            status_line_command: config.status_line_command.clone(),
            status_line_cache: None,
            pending_shell: false,
            keybindings: {
                let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                let settings = crate::config::SettingsManager::load(&cwd).merged;
                std::sync::Arc::new(crate::tui::keybindings::KeybindingMap::from_settings(
                    settings.keybindings.as_ref(),
                ))
            },
            pending_background: false,
            pending_suspend: false,
            exit_when_agent_done: false,
            preserve_chat_on_exit: true,
            last_preserve_chat_setting: true,
            pending_resume_ai: false,
            // SEAL - Session handles preprocessing
            seal_processor: None,
            seal_dialog_state: brainwires_seal::DialogState::new(),
            seal_entity_store: crate::utils::entity_extraction::EntityStore::new(),
            seal_entity_extractor: crate::utils::entity_extraction::EntityExtractor::new(),
            // Multi-Agent System
            pending_agent_switch: None,
            pending_agent_spawn: None,
            ipc_writer: None,  // Will be set by caller
            is_ipc_mode: true, // Always in IPC mode for viewer
            ipc_needs_respawn: false,
            // Skills system
            skill_registry: {
                let mut registry = brainwires_skills::SkillRegistry::new();
                if let Err(e) = crate::utils::skills::discover_skills(&mut registry) {
                    tracing::warn!("Failed to discover skills: {}", e);
                }
                Some(registry)
            },
            pending_skill_tool_scope: None,
            // Journal tree and sub-agent viewer
            journal_tree: super::journal_tree::JournalTreeState::new(),
            sub_agent_viewer_state: None,
            // Plan mode fields
            plan_mode_state: None,
            plan_mode_saved_main: None,
            // PKS integration for implicit fact detection and behavioral inference
            pks_integration: brainwires::knowledge::bks_pks::personal::PksIntegration::default(),
            unread_error_count: 0,
        });

        // Record working directory for PKS after creation (avoids move issue)
        if let Ok(ref mut app) = result {
            app.pks_integration
                .record_working_directory(&app.working_directory);
        }

        result
    }

    /// Create a new application instance (legacy mode - runs AI directly)
    ///
    /// This creates a full App that runs AI provider directly in-process.
    /// Prefer `new_viewer()` for the new architecture where Session handles AI.
    pub async fn new(
        session_id: Option<String>,
        model: Option<String>,
        mdap_config: Option<MdapConfig>,
    ) -> Result<Self> {
        // Load config
        let config_manager = crate::config::ConfigManager::new()?;
        let config = config_manager.get();

        // Use model from config if not provided
        let model = match model {
            Some(m) => m,
            None => config.model.clone(),
        };

        // Load SEAL settings
        let seal_settings = &config.seal;

        // Create provider factory and provider
        let provider_factory = ProviderFactory::new();
        let provider = provider_factory.create(model.clone()).await?;

        // Use simple session ID
        let session_id = session_id
            .unwrap_or_else(|| format!("tui-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S")));

        // Initialize command executor
        let command_executor = CommandExecutor::new()?;

        // Initialize checkpoint manager
        let checkpoint_manager = CheckpointManager::new()?;

        // Initialize prompt history
        let prompt_history = PromptHistory::new()?;

        // Initialize storage
        let db_path = crate::utils::paths::PlatformPaths::conversations_db_path()?;
        let client = std::sync::Arc::new(
            crate::storage::LanceDatabase::new(db_path.to_str().context("Invalid DB path")?)
                .await
                .context("Failed to create LanceDB client")?,
        );
        client
            .initialize(384)
            .await
            .context("Failed to initialize LanceDB")?;
        let embeddings = std::sync::Arc::new(
            crate::storage::embeddings::CachedEmbeddingProvider::new()
                .context("Failed to create embedding provider")?,
        );

        let conversation_store = crate::storage::ConversationStore::new(client.clone());
        let message_store = crate::storage::MessageStore::new(client.clone(), embeddings);
        let task_store = TaskStore::new(client.clone());
        let task_manager = Arc::new(RwLock::new(TaskManager::new()));

        // Initialize session task list (in-memory only, session-specific)
        let session_tasks = Arc::new(RwLock::new(
            crate::types::session_task::SessionTaskList::new(),
        ));

        // Initialize tools and tool executor with core tools only to reduce token cost
        let registry = brainwires_tool_builtins::registry_with_builtins();
        let tools: Vec<_> = registry.get_core().into_iter().cloned().collect();
        let mut tool_executor = ToolExecutor::new(PermissionMode::Auto);

        // Wire task manager with persistence
        tool_executor.set_task_manager_with_persistence(
            Arc::clone(&task_manager),
            task_store.clone(),
            session_id.clone(),
        );

        // Wire session task list
        tool_executor.set_session_task_list(Arc::clone(&session_tasks));

        // Create approval channel and wire it to the executor
        let (approval_tx, approval_rx) = mpsc::channel::<crate::approval::ApprovalRequest>(16);
        tool_executor.set_approval_channel(approval_tx);

        // Create sudo password channel and wire it to the executor
        let (sudo_tx, sudo_rx) = mpsc::channel::<crate::sudo::SudoPasswordRequest>(4);
        tool_executor.set_sudo_password_channel(sudo_tx);

        // Create user-question channel for the `ask_user_question` tool.
        let (user_q_tx, user_q_rx) = mpsc::channel::<crate::ask::UserQuestionRequest>(4);
        tool_executor.set_user_question_channel(user_q_tx);

        let tool_executor = Arc::new(tool_executor);
        let working_directory = std::env::current_dir()?.to_string_lossy().to_string();

        // Build system prompt with CWD context and behavioral knowledge
        let system_prompt = {
            use crate::utils::paths::PlatformPaths;
            use brainwires::knowledge::bks_pks::BehavioralKnowledgeCache;
            use brainwires::knowledge::bks_pks::matcher::{MatchedTruth, format_truths_for_prompt};

            // Try to load BKS cache and get reliable truths
            let truths_section = if let Ok(cache_path) = PlatformPaths::knowledge_db() {
                if let Ok(cache) = BehavioralKnowledgeCache::new(&cache_path, 100) {
                    let reliable = cache.get_reliable_truths(0.5, 30);
                    if !reliable.is_empty() {
                        // Convert to MatchedTruth format (within scope where cache is alive)
                        let matched: Vec<MatchedTruth> = reliable
                            .iter()
                            .map(|t| MatchedTruth {
                                truth: t,
                                match_score: 1.0,
                                effective_confidence: t.decayed_confidence(30),
                            })
                            .collect();
                        format_truths_for_prompt(&matched)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            // Build base prompt and append behavioral knowledge if available
            let base_prompt = build_system_prompt(None)?;
            if truths_section.is_empty() {
                base_prompt
            } else {
                format!("{}\n{}", base_prompt, truths_section)
            }
        };
        let system_message = Message {
            role: Role::System,
            content: MessageContent::Text(system_prompt),
            name: None,
            metadata: None,
        };

        let mut result = Ok(Self {
            session_id,
            messages: Vec::new(),
            conversation_history: vec![system_message],
            input_state: ratatui_interact::components::TextAreaState::empty(),
            input_draft: None,
            scroll: 0,
            active_tools: Vec::new(),
            mode: AppMode::Normal,
            prompt_history,
            search_query: String::new(),
            search_results: Vec::new(),
            search_result_index: 0,
            status: if mdap_config.is_some() {
                format!(
                    "Ready - Model: {} [MDAP] (Ctrl+C to quit, Ctrl+R to search history)",
                    model
                )
            } else {
                format!(
                    "Ready - Model: {} (Ctrl+C to quit, Ctrl+R to search history)",
                    model
                )
            },
            provider,
            model,
            should_quit: false,
            command_executor,
            checkpoint_manager,
            conversation_store,
            message_store,
            cleared_messages: None,
            cleared_conversation_history: None,
            available_sessions: Vec::new(),
            selected_session_index: 0,
            session_picker_scroll: 0,
            console_state: ScrollableContentState::new(vec![
                "Console initialized - Debug messages will appear here".to_string(),
                "Press Ctrl+D to toggle full-screen console view".to_string(),
            ]),
            autocomplete_suggestions: Vec::new(),
            autocomplete_index: 0,
            show_autocomplete: false,
            autocomplete_title: "Slash Commands".to_string(),
            cached_model_ids: Vec::new(),
            active_plan: None,
            completed_plan_steps: Vec::new(),
            approval_mode: ApprovalMode::Suggest, // Default to safest mode
            pending_exec_command: None,
            shell_history: Vec::new(),
            tool_execution_history: Vec::new(),
            selected_shell_index: 0,
            shell_viewer_scroll: 0,
            focused_panel: FocusedPanel::Input, // Start with input focused
            task_manager,
            task_store,
            task_tree_cache: "No tasks".to_string(),
            task_count_cache: 0,
            session_tasks,
            session_task_summary: String::new(),
            session_task_panel_cache: Vec::new(),
            queued_messages: Vec::new(),
            plan_progress: None,
            tools,
            tool_executor,
            working_directory,
            stream_rx: None,
            streaming_content: String::new(),
            streaming_msg_idx: None,
            streaming_conversation: None,
            streaming_user_content: None,
            toast_message: None,
            toast_expires_at: None,
            mouse_capture_disabled: false,
            working_set: crate::types::WorkingSet::new(),
            conversation_area: None,
            input_area: None,
            status_bar_area: None,
            exit_button_area: None,
            status_bar_visible: true,
            last_click_time: None,
            last_click_pos: None,
            conversation_line_count: 0,
            pending_scroll_to_bottom: false,
            scroll_to_bottom_on_resize: false,
            is_pty_session: false,
            prompt_mode: PromptMode::default(),
            pre_plan_prompt_mode: None,
            tool_mode: ToolMode::Smart, // Default to smart routing
            tool_picker_state: None,
            mcp_tools: Vec::new(),
            mcp_connected_servers: Vec::new(),
            tool_rx: None,
            tool_tx: None,
            pending_tool_data: None,
            cancellation_token: None,
            tool_task_handle: None,
            stream_task_handle: None,
            task_viewer_state: TaskViewerState::default(),
            seal_status: SealStatus {
                enabled: seal_settings.enabled,
                last_resolution: None,
                entity_count: 0,
                matched_pattern: None,
                quality_score: 1.0,
                show_status: seal_settings.show_status,
            },
            file_explorer_state: None,
            nano_editor_state: None,
            git_scm_state: None,
            find_replace_state: None,
            mdap_config,
            pending_questions: None,
            question_state: QuestionAnswerState::default(),
            conversation_view_style: ConversationViewStyle::default(),
            help_dialog_state: None,
            suspend_dialog_state: None,
            exit_dialog_state: None,
            hotkey_dialog_state: None,
            approval_dialog_state: None,
            approval_rx: Some(approval_rx),
            sudo_dialog_state: None,
            sudo_password_rx: Some(sudo_rx),
            user_question_rx: Some(user_q_rx),
            active_user_question: None,
            status_line_command: config.status_line_command.clone(),
            status_line_cache: None,
            pending_shell: false,
            keybindings: {
                let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                let settings = crate::config::SettingsManager::load(&cwd).merged;
                std::sync::Arc::new(crate::tui::keybindings::KeybindingMap::from_settings(
                    settings.keybindings.as_ref(),
                ))
            },
            pending_background: false,
            pending_suspend: false,
            exit_when_agent_done: false,
            preserve_chat_on_exit: true,
            last_preserve_chat_setting: true,
            pending_resume_ai: false,
            // SEAL components for query enhancement
            seal_processor: if seal_settings.enabled {
                Some(brainwires_seal::SealProcessor::with_defaults())
            } else {
                None
            },
            seal_dialog_state: brainwires_seal::DialogState::new(),
            seal_entity_store: crate::utils::entity_extraction::EntityStore::new(),
            seal_entity_extractor: crate::utils::entity_extraction::EntityExtractor::new(),
            // Multi-Agent System
            pending_agent_switch: None,
            pending_agent_spawn: None,
            ipc_writer: None,
            is_ipc_mode: false,
            ipc_needs_respawn: false,
            // Skills system
            skill_registry: {
                let mut registry = brainwires_skills::SkillRegistry::new();
                if let Err(e) = crate::utils::skills::discover_skills(&mut registry) {
                    tracing::warn!("Failed to discover skills: {}", e);
                }
                Some(registry)
            },
            pending_skill_tool_scope: None,
            // Journal tree and sub-agent viewer
            journal_tree: super::journal_tree::JournalTreeState::new(),
            sub_agent_viewer_state: None,
            // Plan mode fields
            plan_mode_state: None,
            plan_mode_saved_main: None,
            // PKS integration for implicit fact detection and behavioral inference
            pks_integration: brainwires::knowledge::bks_pks::personal::PksIntegration::default(),
            unread_error_count: 0,
        });

        // Record working directory for PKS after creation (avoids move issue)
        if let Ok(ref mut app) = result {
            app.pks_integration
                .record_working_directory(&app.working_directory);
        }

        result
    }

    /// Get task tree as formatted string for display
    pub async fn get_task_tree_display(&self) -> String {
        let manager = self.task_manager.read().await;
        manager.format_tree().await
    }

    /// Get task count
    pub async fn get_task_count(&self) -> usize {
        let manager = self.task_manager.read().await;
        manager.count().await
    }

    /// Update task tree cache for UI rendering
    pub async fn update_task_cache(&mut self) {
        let manager = self.task_manager.read().await;
        self.task_tree_cache = manager.format_tree().await;
        self.task_count_cache = manager.count().await;
        // Update plan progress
        let stats = manager.get_stats().await;
        if stats.total > 0 {
            self.plan_progress = Some((stats.completed, stats.total));
        }
    }

    /// Update the session task cache for UI rendering
    pub async fn update_session_task_cache(&mut self) {
        let list = self.session_tasks.read().await;
        self.session_task_summary = list.summary();
        self.session_task_panel_cache = list.format_for_panel();
    }

    /// Update SEAL status from processing result
    ///
    /// Call this after SEAL preprocessing to update the UI status display.
    pub fn update_seal_status(&mut self, result: &brainwires_seal::SealProcessingResult) {
        if !self.seal_status.show_status {
            return;
        }

        // Update last resolution if we have any
        if let Some(resolution) = result.resolutions.first() {
            self.seal_status.last_resolution = Some(format!(
                "\"{}\" → {}",
                resolution.reference.text, resolution.antecedent
            ));
        }

        // Update matched pattern
        self.seal_status.matched_pattern = result.matched_pattern.clone();

        // Update quality score
        self.seal_status.quality_score = result.quality_score;
    }

    /// Update SEAL entity count
    pub fn update_seal_entity_count(&mut self, count: usize) {
        self.seal_status.entity_count = count;
    }

    /// Preprocess user input through SEAL pipeline
    ///
    /// This performs coreference resolution and entity extraction to enhance
    /// the user's query before sending to the AI provider.
    ///
    /// Returns the resolved query (with pronouns/references resolved).
    pub fn seal_preprocess(&mut self, user_input: &str) -> String {
        // Extract entities from the input
        let extraction = self.seal_entity_extractor.extract(user_input, "");
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.seal_entity_store
            .add_extraction(extraction, "", timestamp);
        self.seal_status.entity_count = self.seal_entity_store.stats().total_entities;

        // Advance dialog turn
        self.seal_dialog_state.next_turn();

        // Process through SEAL if enabled
        if let Some(ref mut seal) = self.seal_processor {
            match seal.process(
                user_input,
                &self.seal_dialog_state,
                &self.seal_entity_store,
                None, // No relationship graph in TUI context
            ) {
                Ok(result) => {
                    let resolved = result.resolved_query.clone();
                    self.update_seal_status(&result);
                    return resolved;
                }
                Err(e) => {
                    tracing::debug!("SEAL processing error: {:?}", e);
                }
            }
        }

        // Fallback to original input
        user_input.to_string()
    }

    /// Refresh the cached custom status line by running
    /// `status_line_command` if configured. Cached for 1 second to keep
    /// the render loop cheap. Call this from an async context (event loop
    /// tick) — the command runs via `tokio::process::Command` with a
    /// 200 ms timeout; if it takes longer or fails, the previous value
    /// is kept.
    pub async fn refresh_status_line(&mut self) {
        let cmd = match self.status_line_command.as_ref() {
            Some(c) if !c.trim().is_empty() => c.clone(),
            _ => return,
        };
        if let Some((at, _)) = self.status_line_cache
            && at.elapsed() < std::time::Duration::from_millis(1_000)
        {
            return;
        }
        let run = async move {
            use tokio::process::Command;
            let mut child = Command::new("bash");
            child.arg("-c").arg(&cmd);
            child.stdin(std::process::Stdio::null());
            let output = child.output();
            let timed = tokio::time::timeout(std::time::Duration::from_millis(200), output).await;
            match timed {
                Ok(Ok(out)) if out.status.success() => {
                    String::from_utf8_lossy(&out.stdout).trim().to_string()
                }
                _ => String::new(),
            }
        };
        let text = run.await;
        if !text.is_empty() {
            self.status_line_cache = Some((std::time::Instant::now(), text));
        }
    }

    /// Current cached custom status line (or empty).
    pub fn custom_status_line(&self) -> &str {
        self.status_line_cache
            .as_ref()
            .map(|(_, t)| t.as_str())
            .unwrap_or("")
    }

    /// Get SEAL status line for display
    ///
    /// Returns a formatted status line showing SEAL processing information,
    /// or an empty string if SEAL status display is disabled.
    pub fn get_seal_status_line(&self) -> String {
        if !self.seal_status.enabled || !self.seal_status.show_status {
            return String::new();
        }

        let mut parts = Vec::new();

        // Show entity count
        if self.seal_status.entity_count > 0 {
            parts.push(format!("{}📦", self.seal_status.entity_count));
        }

        // Show last resolution
        if let Some(ref resolution) = self.seal_status.last_resolution {
            parts.push(format!("🔮{}", resolution));
        }

        // Show matched pattern
        if let Some(ref pattern) = self.seal_status.matched_pattern {
            parts.push(format!("📚{}", pattern));
        }

        if parts.is_empty() {
            String::new()
        } else {
            format!("[SEAL: {}]", parts.join(" | "))
        }
    }

    /// Get display status
    pub fn get_status(&self) -> String {
        use super::history::HistoryOps;
        match self.mode {
            AppMode::Normal => {
                // Add plan progress to normal status if available
                let base_status = if let (Some(plan), Some((completed, total))) =
                    (&self.active_plan, &self.plan_progress)
                {
                    let progress_pct = if *total > 0 {
                        (*completed * 100) / *total
                    } else {
                        0
                    };
                    format!(
                        "Plan: {} ({}/{} tasks, {}%)",
                        plan.title.chars().take(30).collect::<String>(),
                        completed,
                        total,
                        progress_pct
                    )
                } else if let Some(plan) = &self.active_plan {
                    format!("Plan: {}", plan.title.chars().take(40).collect::<String>())
                } else {
                    self.status.clone()
                };

                // Append SEAL status if available
                let seal_status = self.get_seal_status_line();
                let mut assembled = if seal_status.is_empty() {
                    base_status
                } else {
                    format!("{} {}", base_status, seal_status)
                };
                // Append user-configured custom status line (cached).
                let custom = self.custom_status_line();
                if !custom.is_empty() {
                    assembled.push(' ');
                    assembled.push_str(custom);
                }
                assembled
            }
            AppMode::ReverseSearch => {
                let current_result = self.get_current_search_result().unwrap_or_default();
                let match_info = if !self.search_results.is_empty() {
                    format!(
                        " [{}/{}]",
                        self.search_result_index + 1,
                        self.search_results.len()
                    )
                } else {
                    String::new()
                };
                format!(
                    "(reverse-i-search)`{}'{}:  {}",
                    self.search_query, match_info, current_result
                )
            }
            AppMode::SessionPicker => "Select session...".to_string(),
            AppMode::ConsoleView => format!(
                "Console - {} messages (Esc to exit)",
                self.console_state.line_count()
            ),
            AppMode::ShellViewer => format!(
                "Shell History - {} commands (Esc to exit)",
                self.shell_history.len()
            ),
            AppMode::ConversationFullscreen => format!(
                "Conversation - {} messages (Esc to exit)",
                self.messages.len()
            ),
            AppMode::InputFullscreen => "Input (Fullscreen) - Esc to exit".to_string(),
            AppMode::Waiting => {
                // Show plan progress while waiting
                if let Some((completed, total)) = &self.plan_progress {
                    format!("Working on plan... ({}/{} tasks)", completed, total)
                } else if self.active_plan.is_some() {
                    "Working on plan...".to_string()
                } else {
                    "Waiting for response...".to_string()
                }
            }
            AppMode::ToolPicker => {
                "Select tools... (Space: toggle, Enter: confirm, Esc: cancel)".to_string()
            }
            AppMode::TaskViewer => {
                format!("Task Tree - {} tasks (Esc to close)", self.task_count_cache)
            }
            AppMode::FileExplorer => {
                "File Explorer (Esc to close, Enter to open, Space to select)".to_string()
            }
            AppMode::NanoEditor => "Nano Editor (Ctrl+S to save, Ctrl+X to exit)".to_string(),
            AppMode::GitScm => {
                "Git SCM (Tab: panels, s: stage, u: unstage, c: commit, Esc: close)".to_string()
            }
            AppMode::CancelConfirm => "Cancel operation? (y/n)".to_string(),
            AppMode::QuestionAnswer => {
                if let Some(ref questions) = self.pending_questions {
                    let current = self.question_state.current_question_idx + 1;
                    let total = questions.questions.len();
                    format!(
                        "Clarifying Questions ({}/{}) - ↑↓: Navigate, Space: Select, Tab: Next, Esc: Skip",
                        current, total
                    )
                } else {
                    "Clarifying Questions".to_string()
                }
            }
            AppMode::FindDialog => {
                if let Some(ref state) = self.find_replace_state {
                    let status = state.status_message.as_deref().unwrap_or("");
                    format!("Find: {} (F3/Shift+F3: next/prev, Esc: close)", status)
                } else {
                    "Find".to_string()
                }
            }
            AppMode::FindReplaceDialog => {
                if let Some(ref state) = self.find_replace_state {
                    let status = state.status_message.as_deref().unwrap_or("");
                    format!(
                        "Find & Replace: {} (Tab: switch, Enter: replace, Esc: close)",
                        status
                    )
                } else {
                    "Find & Replace".to_string()
                }
            }
            AppMode::HelpDialog => {
                "Help (F1/Esc to close, Tab: switch panel, ↑/↓: navigate)".to_string()
            }
            AppMode::SuspendDialog => {
                "Ctrl+Z: Background or Suspend? (Tab: switch, Enter: select, Esc: cancel)"
                    .to_string()
            }
            AppMode::ExitDialog => {
                "Ctrl+C: Exit or Background? (Tab: switch, Enter: select, Esc: cancel)".to_string()
            }
            AppMode::HotkeyDialog => {
                "Hotkey Configuration (Tab: switch panel, ↑/↓: navigate, Esc: close)".to_string()
            }
            AppMode::ApprovalDialog => {
                "Tool Approval ([Y]es / [N]o / [A]lways / [D]eny always)".to_string()
            }
            AppMode::SudoPasswordDialog => "Sudo Password (Enter: submit, Esc: cancel)".to_string(),
            AppMode::UserQuestion => {
                "Agent Question (↑↓: navigate, Space: select, Enter: submit, Esc: cancel)"
                    .to_string()
            }
            AppMode::PlanMode => {
                let msg_count = self
                    .plan_mode_state
                    .as_ref()
                    .map(|s| s.message_count())
                    .unwrap_or(0);
                format!("Plan Mode - {} messages (Ctrl+P to exit)", msg_count)
            }
            AppMode::SubAgentViewer => {
                "Sub-Agent Viewer (Tab: switch panel, Esc: close)".to_string()
            }
        }
    }

    /// Add a message to the console
    pub fn add_console_message(&mut self, msg: String) {
        self.console_state.push_line(msg);
    }

    /// Set the status bar text AND permanently journal the event with timestamp + level.
    ///
    /// Use this instead of bare `self.status = ...` so nothing gets silently lost.
    pub fn set_status(&mut self, level: LogLevel, msg: impl Into<String>) {
        let msg = msg.into();
        self.status = msg.clone();
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        self.add_console_message(format!("{} [{}] {}", ts, level.label(), msg));
        if matches!(level, LogLevel::Warn | LogLevel::Error) {
            self.unread_error_count += 1;
        }
    }

    /// Clear the unread error/warning badge (call when the user opens the journal).
    pub fn clear_unread_errors(&mut self) {
        self.unread_error_count = 0;
    }

    /// Record a tool execution for Journal display
    ///
    /// This adds a tool execution entry to the history, which will be displayed
    /// interleaved with messages in the Journal view.
    pub fn record_tool_execution(
        &mut self,
        tool_name: &str,
        parameters: &serde_json::Value,
        result: Option<&str>,
        success: bool,
        duration_ms: Option<u64>,
    ) {
        // Create a brief summary of parameters
        let params_summary = Self::summarize_tool_params(tool_name, parameters);

        // Create a brief summary of the result
        let result_summary = result
            .map(|r| Self::truncate_for_display(r, 150))
            .unwrap_or_else(|| "(no output)".to_string());

        let entry = ToolExecutionEntry {
            tool_name: tool_name.to_string(),
            parameters_summary: params_summary,
            result_summary,
            success,
            executed_at: chrono::Utc::now().timestamp(),
            duration_ms,
        };

        self.tool_execution_history.push(entry);
    }

    /// Create a brief summary of tool parameters for display
    fn summarize_tool_params(tool_name: &str, params: &serde_json::Value) -> String {
        match tool_name {
            // File operations: show the path
            "read_file" | "write_file" | "edit_file" | "delete_file" => params
                .get("path")
                .and_then(|v| v.as_str())
                .map(|s| Self::truncate_for_display(s, 60))
                .unwrap_or_else(|| "(unknown path)".to_string()),
            // Bash: show the command
            "bash" | "execute_command" => params
                .get("command")
                .and_then(|v| v.as_str())
                .map(|s| Self::truncate_for_display(s, 80))
                .unwrap_or_else(|| "(command)".to_string()),
            // Search/glob: show the pattern
            "glob" | "search" | "grep" => params
                .get("pattern")
                .and_then(|v| v.as_str())
                .map(|s| format!("pattern: {}", Self::truncate_for_display(s, 50)))
                .unwrap_or_else(|| "(search)".to_string()),
            // Web fetch: show the URL
            "web_fetch" | "fetch_url" => params
                .get("url")
                .and_then(|v| v.as_str())
                .map(|s| Self::truncate_for_display(s, 60))
                .unwrap_or_else(|| "(url)".to_string()),
            // Default: show compact JSON or first key=value
            _ => {
                if let Some(obj) = params.as_object() {
                    if obj.is_empty() {
                        "(no params)".to_string()
                    } else if obj.len() == 1 {
                        if let Some((k, v)) = obj.iter().next() {
                            let v_str = match v {
                                serde_json::Value::String(s) => Self::truncate_for_display(s, 40),
                                _ => Self::truncate_for_display(&v.to_string(), 40),
                            };
                            format!("{}: {}", k, v_str)
                        } else {
                            "...".to_string()
                        }
                    } else {
                        // Multiple params: show count
                        format!("{} params", obj.len())
                    }
                } else {
                    Self::truncate_for_display(&params.to_string(), 50)
                }
            }
        }
    }

    /// Truncate a string for display, adding ellipsis if needed
    fn truncate_for_display(s: &str, max_len: usize) -> String {
        // Remove newlines for single-line display
        let single_line: String = s
            .chars()
            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
            .collect();

        if single_line.len() <= max_len {
            single_line
        } else {
            format!("{}...", &single_line[..max_len.saturating_sub(3)])
        }
    }

    /// Get the number of queued messages
    pub fn queued_message_count(&self) -> usize {
        self.queued_messages.len()
    }

    /// Show a toast notification for a duration (in milliseconds)
    pub fn show_toast(&mut self, message: String, duration_ms: i64) {
        let now = chrono::Utc::now().timestamp_millis();
        self.toast_message = Some(message);
        self.toast_expires_at = Some(now + duration_ms);
    }

    /// Get the current toast message if not expired
    pub fn get_toast(&self) -> Option<&str> {
        if let (Some(msg), Some(expires)) = (&self.toast_message, self.toast_expires_at) {
            let now = chrono::Utc::now().timestamp_millis();
            if now < expires {
                return Some(msg.as_str());
            }
        }
        None
    }

    /// Clear expired toast
    pub fn clear_expired_toast(&mut self) {
        if let Some(expires) = self.toast_expires_at {
            let now = chrono::Utc::now().timestamp_millis();
            if now >= expires {
                self.toast_message = None;
                self.toast_expires_at = None;
            }
        }
    }

    /// Transition to Normal mode after streaming/processing completes.
    /// This only changes the mode if we're currently in a streaming-related mode
    /// (Waiting, CancelConfirm). Dialog modes like ExitDialog, SuspendDialog, etc.
    /// are preserved so they don't get closed unexpectedly.
    pub fn transition_to_normal_after_streaming(&mut self) {
        if matches!(self.mode, AppMode::Waiting | AppMode::CancelConfirm) {
            self.mode = AppMode::Normal;
        }
    }

    /// Check if a point (column, row) is within the conversation panel area
    pub fn is_point_in_conversation_area(&self, col: u16, row: u16) -> bool {
        if let Some(area) = self.conversation_area {
            col >= area.x
                && col < area.x + area.width
                && row >= area.y
                && row < area.y + area.height
        } else {
            // If we don't have the area cached, default to true (old behavior)
            true
        }
    }

    /// Check if a point (column, row) is within the input panel area
    pub fn is_point_in_input_area(&self, col: u16, row: u16) -> bool {
        if let Some(area) = self.input_area {
            col >= area.x
                && col < area.x + area.width
                && row >= area.y
                && row < area.y + area.height
        } else {
            false
        }
    }

    /// Check if a point (column, row) is within the exit button area
    pub fn is_point_in_exit_button(&self, col: u16, row: u16) -> bool {
        if let Some(area) = self.exit_button_area {
            col >= area.x
                && col < area.x + area.width
                && row >= area.y
                && row < area.y + area.height
        } else {
            false
        }
    }

    /// Double-click detection threshold in milliseconds
    const DOUBLE_CLICK_THRESHOLD_MS: i64 = 500;
    /// Maximum distance (in cells) for double-click detection
    const DOUBLE_CLICK_MAX_DISTANCE: u16 = 3;

    /// Check if a click at (col, row) is a double-click based on timing and position.
    /// Returns true if this is a double-click, false otherwise.
    /// Also records this click for future double-click detection.
    pub fn check_double_click(&mut self, col: u16, row: u16) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let is_double = if let (Some(last_time), Some((last_col, last_row))) =
            (self.last_click_time, self.last_click_pos)
        {
            let time_diff = now - last_time;
            let col_diff = (col as i32 - last_col as i32).unsigned_abs() as u16;
            let row_diff = (row as i32 - last_row as i32).unsigned_abs() as u16;

            time_diff <= Self::DOUBLE_CLICK_THRESHOLD_MS
                && col_diff <= Self::DOUBLE_CLICK_MAX_DISTANCE
                && row_diff <= Self::DOUBLE_CLICK_MAX_DISTANCE
        } else {
            false
        };

        if is_double {
            // Reset after double-click to prevent triple-click being detected as another double
            self.last_click_time = None;
            self.last_click_pos = None;
        } else {
            // Record this click for potential double-click detection
            self.last_click_time = Some(now);
            self.last_click_pos = Some((col, row));
        }

        is_double
    }

    /// Calculate maximum scroll value using the cached line count from render
    pub fn max_scroll(&self) -> u16 {
        let visible_height = self
            .conversation_area
            .map(|a| a.height.saturating_sub(2) as usize) // -2 for borders
            .unwrap_or(20);

        // Only scroll if content exceeds viewport
        if self.conversation_line_count <= visible_height {
            return 0; // Content fits, no scrolling needed
        }

        // max_scroll = how many lines are hidden above when scrolled to bottom
        (self
            .conversation_line_count
            .saturating_sub(visible_height)
            .saturating_sub(1)) as u16
    }

    /// Scroll the conversation view to the bottom
    pub fn scroll_to_bottom(&mut self) {
        self.scroll = self.max_scroll();
    }

    /// Scroll up by amount, clamped to bounds
    pub fn scroll_up(&mut self, amount: u16) {
        self.scroll = self.scroll.saturating_sub(amount);
    }

    /// Scroll down by amount, clamped to max
    pub fn scroll_down(&mut self, amount: u16) {
        let max = self.max_scroll();
        self.scroll = self.scroll.saturating_add(amount).min(max);
    }

    /// Estimated visible height of the conversation panel (lines).
    /// Used for scroll-to-cursor calculations. Falls back to a sensible default
    /// when we don't yet have a measured area.
    pub fn conversation_visible_height(&self) -> u16 {
        self.conversation_area
            .map(|a| a.height.saturating_sub(2)) // subtract borders
            .unwrap_or(20)
    }

    /// Refresh the sub-agent list from TaskManager for the Sub-Agent Viewer.
    pub async fn refresh_sub_agent_list(&mut self) {
        // get_all_tasks() is an async method on TaskManager itself (not on the guard)
        let tasks = self.task_manager.read().await.get_all_tasks().await;

        let sessions_dir = crate::utils::paths::PlatformPaths::sessions_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"));

        let mut agent_list: Vec<SubAgentSummary> = Vec::new();
        for task in tasks {
            // Only show tasks that are assigned to an agent
            let agent_id = match task.assigned_to.clone() {
                Some(id) => id,
                None => continue,
            };

            let task_desc = {
                let s = task.description.as_str();
                if s.len() > 60 {
                    format!("{}…", &s[..57])
                } else {
                    s.to_string()
                }
            };

            // Check for IPC socket
            let sock_path = sessions_dir.join(format!("{}.sock", agent_id));
            let has_ipc_socket = sock_path.exists();

            // Derive a simplified status from task status
            use crate::types::agent::TaskStatus;
            let status = match task.status {
                TaskStatus::InProgress => TaskAgentStatus::Working(task_desc.clone()),
                TaskStatus::Completed => TaskAgentStatus::Completed(
                    task.summary.clone().unwrap_or_else(|| "Done".to_string()),
                ),
                TaskStatus::Failed => TaskAgentStatus::Failed("Task failed".to_string()),
                TaskStatus::Blocked => TaskAgentStatus::Paused("Blocked on dependency".to_string()),
                TaskStatus::Skipped => TaskAgentStatus::Completed("Skipped".to_string()),
                TaskStatus::Pending => TaskAgentStatus::Idle,
            };

            agent_list.push(SubAgentSummary {
                agent_id,
                task_desc,
                status,
                iterations: task.iterations,
                session_id: None,
                has_ipc_socket,
            });
        }

        if let Some(state) = &mut self.sub_agent_viewer_state {
            state.agent_list = agent_list;
        } else {
            self.sub_agent_viewer_state = Some(SubAgentViewerState {
                agent_list,
                selected_index: 0,
                list_scroll: 0,
                detail_scroll: 0,
                message_input: String::new(),
                panel_focus: SubAgentPanelFocus::Left,
            });
        }
    }

    // === Input convenience accessors (delegates to TextAreaState) ===

    /// Get the current input text.
    pub fn input_text(&self) -> String {
        self.input_state.text()
    }

    /// Check if input is empty.
    pub fn input_is_empty(&self) -> bool {
        self.input_state.is_empty()
    }

    /// Check if input contains multiple lines.
    pub fn is_multiline_input(&self) -> bool {
        self.input_state.line_count() > 1
    }

    /// Check if cursor is on the first line.
    pub fn cursor_on_first_line(&self) -> bool {
        self.input_state.cursor_line == 0
    }

    /// Check if cursor is on the last line.
    pub fn cursor_on_last_line(&self) -> bool {
        self.input_state.cursor_line == self.input_state.lines.len().saturating_sub(1)
    }

    /// Clear input and reset cursor.
    pub fn clear_input(&mut self) {
        self.input_state.clear();
    }

    /// Handle pasted text (from bracketed paste mode or rapid-input detection).
    /// Inserts the text at the cursor position and shows a toast with line count.
    pub fn handle_paste(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        // Normalize line endings: convert \r\n to \n, then lone \r to \n
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");

        // Count lines for feedback
        let line_count = normalized.lines().count().max(1);
        let char_count = normalized.chars().count();

        // Insert the normalized text at cursor position
        self.input_state.insert_str(&normalized);

        // Show toast with paste info
        if line_count > 1 {
            self.show_toast(format!("Pasted {} lines", line_count), 2000);
        } else if char_count > 50 {
            self.show_toast(format!("Pasted {} chars", char_count), 1500);
        }
        // For short single-line pastes, don't show toast (behaves like normal typing)
    }

    /// Initialize MCP servers - auto-connect to all configured servers
    pub async fn initialize_mcp(&mut self) {
        use crate::mcp::{McpClient, McpConfigManager, McpToolAdapter};

        // Load configured servers
        let config = match McpConfigManager::load() {
            Ok(c) => c,
            Err(e) => {
                self.add_console_message(format!("⚠️ Failed to load MCP config: {}", e));
                return;
            }
        };

        let servers = config.get_servers();
        if servers.is_empty() {
            return;
        }

        self.add_console_message(format!(
            "🔌 Connecting to {} MCP server(s)...",
            servers.len()
        ));

        // Create a shared MCP client
        let client = Arc::new(RwLock::new(McpClient::new(
            "brainwires",
            env!("CARGO_PKG_VERSION"),
        )));

        // Auto-connect to all configured servers
        for server_config in servers {
            let server_name = server_config.name.clone();

            {
                let client_guard = client.write().await;
                match client_guard.connect(server_config).await {
                    Ok(_) => {
                        // Connection successful, now list tools
                        match client_guard.list_tools(&server_name).await {
                            Ok(mcp_tools) => {
                                let tool_count = mcp_tools.len();

                                // Convert MCP tools to our Tool format
                                let adapter =
                                    McpToolAdapter::new(client.clone(), server_name.clone());

                                match adapter.get_tools().await {
                                    Ok(tools) => {
                                        self.mcp_tools.extend(tools);
                                        self.mcp_connected_servers.push(server_name.clone());
                                        self.add_console_message(format!(
                                            "✅ MCP server '{}' connected ({} tools)",
                                            server_name, tool_count
                                        ));
                                    }
                                    Err(e) => {
                                        self.add_console_message(format!(
                                            "⚠️ Failed to convert tools from '{}': {}",
                                            server_name, e
                                        ));
                                    }
                                }
                            }
                            Err(e) => {
                                self.add_console_message(format!(
                                    "⚠️ Failed to list tools from '{}': {}",
                                    server_name, e
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        self.add_console_message(format!(
                            "⚠️ Failed to connect to MCP server '{}': {}",
                            server_name, e
                        ));
                    }
                }
            }
        }

        let total_tools = self.mcp_tools.len();
        let connected = self.mcp_connected_servers.len();
        if connected > 0 {
            self.add_console_message(format!(
                "📦 MCP initialized: {} server(s), {} tool(s) available",
                connected, total_tools
            ));
        }
    }
}
