//! Agent State
//!
//! Contains the core agent state that persists across TUI attach/detach cycles.
//! This state is held by the Agent process and synchronized to viewers via IPC.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::agent::message_queue::MessageQueue;
use crate::agents::TaskManager;
use crate::commands::CommandExecutor;
use crate::mdap::MdapConfig;
use crate::providers::{Provider, ProviderFactory};
use crate::storage::{LockStore, TaskStore, VectorDatabase};
use crate::tools::ToolExecutor;
use crate::types::agent::PermissionMode;
use crate::types::message::{Message, MessageContent, Role};
use crate::types::plan_mode::{PlanModeState, SavedMainContext};
use crate::types::tool::{Tool, ToolMode};
use crate::utils::checkpoint::CheckpointManager;
use crate::utils::paths::PlatformPaths;
use crate::utils::system_prompt::build_system_prompt;
use brainwires::agent_network::ipc::{AgentMessage, DisplayMessage};
use brainwires::knowledge::bks_pks::{
    BehavioralKnowledgeCache, LearningCollector, detect_correction,
};

/// Core agent state that persists across viewer attach/detach cycles
pub struct AgentState {
    /// Session ID
    pub session_id: String,

    /// Display messages for TUI (simplified format)
    pub messages: Vec<DisplayMessage>,

    /// Full conversation history for API calls
    pub conversation_history: Vec<Message>,

    /// Current status message
    pub status: String,

    /// AI Provider
    pub provider: Arc<dyn Provider>,

    /// Model name
    pub model: String,

    /// Command executor
    pub command_executor: CommandExecutor,

    /// Checkpoint manager
    pub checkpoint_manager: CheckpointManager,

    /// Conversation store
    pub conversation_store: crate::storage::ConversationStore,

    /// Message store
    pub message_store: crate::storage::MessageStore,

    /// Task manager for tracking tasks
    pub task_manager: Arc<RwLock<TaskManager>>,

    /// Task store for persistence
    pub task_store: TaskStore,

    /// Task tree cache (formatted string)
    pub task_tree_cache: String,

    /// Task count cache
    pub task_count_cache: usize,

    /// Available tools for AI
    pub tools: Vec<Tool>,

    /// Tool executor
    pub tool_executor: Arc<ToolExecutor>,

    /// Current working directory
    pub working_directory: String,

    /// Current tool selection mode
    pub tool_mode: ToolMode,

    /// MCP tools from connected servers
    pub mcp_tools: Vec<Tool>,

    /// Connected MCP server names
    pub mcp_connected_servers: Vec<String>,

    /// MDAP configuration (if enabled)
    pub mdap_config: Option<MdapConfig>,

    /// Cancellation token for current operation
    pub cancellation_token: Option<CancellationToken>,

    /// Working set of files in context
    pub working_set: crate::types::WorkingSet,

    /// Whether an operation is in progress
    pub is_busy: bool,

    /// Accumulated response content during streaming
    pub streaming_content: String,

    /// If true, exit when agent work completes
    pub exit_when_done: bool,

    /// If true, there's a pending user request that needs AI response
    pub has_pending_request: bool,

    /// SEAL processor for query enhancement
    pub seal_processor: Option<brainwires_seal::SealProcessor>,

    /// Dialog state for SEAL
    pub seal_dialog_state: brainwires_seal::DialogState,

    /// Entity store for SEAL
    pub seal_entity_store: crate::utils::entity_extraction::EntityStore,

    /// Entity extractor for SEAL
    pub seal_entity_extractor: crate::utils::entity_extraction::EntityExtractor,

    /// SEAL status
    pub seal_enabled: bool,

    /// Lock store for inter-process coordination
    pub lock_store: Arc<LockStore>,

    /// Message queue for pending injections
    pub message_queue: MessageQueue,

    /// Learning collector for implicit learning
    pub learning_collector: LearningCollector,

    /// BKS cache for storing learned truths
    pub bks_cache: Option<BehavioralKnowledgeCache>,

    // Plan Mode Fields
    /// Plan mode state (when active)
    pub plan_mode_state: Option<PlanModeState>,

    /// Saved main context when entering plan mode (for restoration)
    pub saved_main_context: Option<SavedMainContext>,

    /// Whether currently in plan mode
    pub is_plan_mode: bool,
}

impl AgentState {
    /// Create a new agent state
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

        // Generate session ID
        let session_id = session_id
            .unwrap_or_else(|| format!("agent-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S")));

        // Initialize command executor
        let command_executor = CommandExecutor::new()?;

        // Initialize checkpoint manager
        let checkpoint_manager = CheckpointManager::new()?;

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

        // Initialize lock store for inter-process coordination
        let lock_store = Arc::new(
            LockStore::new_default()
                .await
                .context("Failed to create lock store")?,
        );

        // Initialize tools with core tools only
        let registry = brainwires_tool_builtins::registry_with_builtins();
        let tools: Vec<_> = registry.get_core().into_iter().cloned().collect();
        let mut tool_executor = ToolExecutor::new(PermissionMode::Auto);

        // Wire task manager with persistence
        tool_executor.set_task_manager_with_persistence(
            Arc::clone(&task_manager),
            task_store.clone(),
            session_id.clone(),
        );

        // Attach layered settings (permissions) + hook dispatcher so
        // PreToolUse / PostToolUse hooks fire around every tool call.
        {
            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            let (settings, hooks) = crate::hooks::load_for_cwd(&cwd);
            tool_executor.set_settings(settings);
            tool_executor.set_hooks(hooks);
        }

        let tool_executor = Arc::new(tool_executor);
        let working_directory = std::env::current_dir()?.to_string_lossy().to_string();

        // Build system prompt
        let system_prompt = {
            use crate::utils::paths::PlatformPaths;
            use brainwires::knowledge::bks_pks::BehavioralKnowledgeCache;
            use brainwires::knowledge::bks_pks::matcher::{MatchedTruth, format_truths_for_prompt};

            let truths_section = if let Ok(cache_path) = PlatformPaths::knowledge_db() {
                if let Ok(cache) = BehavioralKnowledgeCache::new(&cache_path, 100) {
                    let reliable = cache.get_reliable_truths(0.5, 30);
                    if !reliable.is_empty() {
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

        // Try to load existing conversation from storage
        let (mut messages, mut conversation_history) = (Vec::new(), vec![system_message]);
        let mut has_pending_request = false;

        // Load messages from storage for existing session
        if let Ok(stored_messages) = message_store.get_by_conversation(&session_id).await
            && !stored_messages.is_empty()
        {
            tracing::info!(
                "Loading {} messages from storage for session {}",
                stored_messages.len(),
                session_id
            );
            for msg in stored_messages {
                // Add to display messages
                messages.push(DisplayMessage::new(&msg.role, &msg.content, msg.created_at));

                // Add to conversation history
                let role = match msg.role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => continue, // Skip system messages from storage (we already have one)
                };
                conversation_history.push(Message {
                    role,
                    content: MessageContent::Text(msg.content),
                    name: None,
                    metadata: None,
                });
            }

            // Check if last message is from user (pending request)
            // This happens when user backgrounded before AI could respond
            if let Some(last_msg) = messages.last()
                && last_msg.role == "user"
            {
                has_pending_request = true;
                tracing::info!("Detected pending user request that needs AI response");
            }
        }

        Ok(Self {
            session_id,
            messages,
            conversation_history,
            status: format!("Ready - Model: {}", model),
            provider,
            model,
            command_executor,
            checkpoint_manager,
            conversation_store,
            message_store,
            task_manager,
            task_store,
            task_tree_cache: "No tasks".to_string(),
            task_count_cache: 0,
            tools,
            tool_executor,
            working_directory,
            tool_mode: ToolMode::Smart,
            mcp_tools: Vec::new(),
            mcp_connected_servers: Vec::new(),
            mdap_config,
            cancellation_token: None,
            working_set: crate::types::WorkingSet::new(),
            is_busy: false,
            streaming_content: String::new(),
            exit_when_done: false,
            has_pending_request,
            seal_processor: if seal_settings.enabled {
                Some(brainwires_seal::SealProcessor::with_defaults())
            } else {
                None
            },
            seal_dialog_state: brainwires_seal::DialogState::new(),
            seal_entity_store: crate::utils::entity_extraction::EntityStore::new(),
            seal_entity_extractor: crate::utils::entity_extraction::EntityExtractor::new(),
            seal_enabled: seal_settings.enabled,
            lock_store,
            message_queue: MessageQueue::default(),
            learning_collector: LearningCollector::new(3, None), // 3 occurrences to trigger pattern
            bks_cache: PlatformPaths::knowledge_db()
                .ok()
                .and_then(|db_path| BehavioralKnowledgeCache::new(&db_path, 100).ok()),
            // Plan mode fields
            plan_mode_state: None,
            saved_main_context: None,
            is_plan_mode: false,
        })
    }

    /// Add a user message to the conversation
    pub fn add_user_message(&mut self, content: String) {
        let timestamp = chrono::Utc::now().timestamp_millis();

        // Check for corrections in the user message (implicit learning)
        self.analyze_for_corrections(&content);

        // Add to display messages
        self.messages
            .push(DisplayMessage::new("user", &content, timestamp));

        // Add to conversation history
        self.conversation_history.push(Message::user(&content));
    }

    /// Add an assistant message to the conversation
    pub fn add_assistant_message(&mut self, content: String) {
        let timestamp = chrono::Utc::now().timestamp_millis();

        // Add to display messages
        self.messages
            .push(DisplayMessage::new("assistant", &content, timestamp));

        // Add to conversation history
        self.conversation_history.push(Message::assistant(&content));
    }

    /// Analyze user message for corrections (implicit learning)
    fn analyze_for_corrections(&mut self, user_message: &str) {
        // Get the last assistant message for context
        let last_assistant_msg = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        // Try to detect correction patterns
        if let Some((wrong, right)) = detect_correction(user_message) {
            let context = if last_assistant_msg.len() > 100 {
                last_assistant_msg[..100].to_string()
            } else {
                last_assistant_msg.clone()
            };

            // Record the correction
            self.learning_collector
                .record_correction(&context, &wrong, &right);

            tracing::info!(
                "Detected implicit correction: wrong='{}' -> right='{}'",
                &wrong[..wrong.len().min(30)],
                &right[..right.len().min(30)]
            );

            // Process signals and store learned truths
            self.process_learning_signals();
        }
    }

    /// Record a tool outcome for learning
    pub fn record_tool_outcome(
        &mut self,
        tool_name: &str,
        command: &str,
        success: bool,
        error_message: Option<&str>,
        execution_time_ms: u64,
    ) {
        self.learning_collector.record_tool_outcome(
            tool_name,
            command,
            success,
            error_message,
            execution_time_ms,
        );

        // Only process signals periodically or on failures
        if !success {
            self.process_learning_signals();
        }
    }

    /// Process queued learning signals and store them to BKS
    fn process_learning_signals(&mut self) {
        let truths = self.learning_collector.process_signals();

        if truths.is_empty() {
            return;
        }

        tracing::info!("Processing {} learning signals into BKS", truths.len());

        if let Some(ref mut bks_cache) = self.bks_cache {
            for truth in truths {
                if let Err(e) = bks_cache.add_truth(truth) {
                    tracing::warn!("Failed to add learned truth to BKS: {}", e);
                }
            }
        }
    }

    /// Update task cache
    pub async fn update_task_cache(&mut self) {
        let manager = self.task_manager.read().await;
        self.task_tree_cache = manager.format_tree().await;
        self.task_count_cache = manager.count().await;
    }

    /// Create a ConversationSync message for IPC
    pub fn create_sync_message(&self) -> AgentMessage {
        AgentMessage::ConversationSync {
            session_id: self.session_id.clone(),
            model: self.model.clone(),
            messages: self.messages.clone(),
            status: self.status.clone(),
            is_busy: self.is_busy,
            tool_mode: self.tool_mode.clone(),
            mcp_servers: self.mcp_connected_servers.clone(),
        }
    }

    /// Create a TaskUpdate message for IPC
    pub fn create_task_update_message(&self) -> AgentMessage {
        let completed_count = self
            .task_tree_cache
            .lines()
            .filter(|l| l.contains("✓"))
            .count();

        AgentMessage::TaskUpdate {
            task_tree: self.task_tree_cache.clone(),
            task_count: self.task_count_cache,
            completed_count,
        }
    }

    /// Preprocess user input through SEAL pipeline
    pub fn seal_preprocess(&mut self, user_input: &str) -> String {
        // Extract entities
        let extraction = self.seal_entity_extractor.extract(user_input, "");
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.seal_entity_store
            .add_extraction(extraction, "", timestamp);

        // Advance dialog turn
        self.seal_dialog_state.next_turn();

        // Process through SEAL if enabled
        if let Some(ref mut seal) = self.seal_processor {
            match seal.process(
                user_input,
                &self.seal_dialog_state,
                &self.seal_entity_store,
                None,
            ) {
                Ok(result) => {
                    return result.resolved_query;
                }
                Err(e) => {
                    tracing::debug!("SEAL processing error: {:?}", e);
                }
            }
        }

        user_input.to_string()
    }

    /// Get SEAL status for IPC
    pub fn get_seal_status(&self) -> AgentMessage {
        AgentMessage::SealStatus {
            enabled: self.seal_enabled,
            entity_count: self.seal_entity_store.stats().total_entities,
            last_resolution: None, // TODO: track this
            quality_score: 1.0,
        }
    }

    /// Initialize MCP servers
    pub async fn initialize_mcp(&mut self) {
        use crate::mcp::{McpClient, McpConfigManager, McpToolAdapter};

        let config = match McpConfigManager::load() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to load MCP config: {}", e);
                return;
            }
        };

        let servers = config.get_servers();
        if servers.is_empty() {
            return;
        }

        tracing::info!("Connecting to {} MCP server(s)...", servers.len());

        let client = Arc::new(RwLock::new(McpClient::new(
            "brainwires",
            env!("CARGO_PKG_VERSION"),
        )));

        for server_config in servers {
            let server_name = server_config.name.clone();

            let client_guard = client.write().await;
            match client_guard.connect(server_config).await {
                Ok(_) => match client_guard.list_tools(&server_name).await {
                    Ok(mcp_tools) => {
                        let tool_count = mcp_tools.len();

                        let adapter = McpToolAdapter::new(client.clone(), server_name.clone());

                        match adapter.get_tools().await {
                            Ok(tools) => {
                                self.mcp_tools.extend(tools);
                                self.mcp_connected_servers.push(server_name.clone());
                                tracing::info!(
                                    "MCP server '{}' connected ({} tools)",
                                    server_name,
                                    tool_count
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to convert tools from '{}': {}",
                                    server_name,
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to list tools from '{}': {}", server_name, e);
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to connect to MCP server '{}': {}", server_name, e);
                }
            }
        }

        let total_tools = self.mcp_tools.len();
        let connected = self.mcp_connected_servers.len();
        if connected > 0 {
            tracing::info!(
                "MCP initialized: {} server(s), {} tool(s) available",
                connected,
                total_tools
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_message_new() {
        let msg = DisplayMessage::new("user", "Hello", 1234567890);
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
        assert_eq!(msg.created_at, 1234567890);
    }
}
