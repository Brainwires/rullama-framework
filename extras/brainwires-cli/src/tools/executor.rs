use anyhow::Result;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    AgentPoolTool, AskUserQuestionTool, BashTool, CodeExecTool, ContextRecallTool, FileOpsTool,
    GitTool, McpToolExecutor, MemoryTool, MonitorTool, OrchestratorTool, PlanTool, SearchTool,
    SemanticSearchTool, SessionTaskTool, TaskManagerTool, ToolRegistry, ToolSearchTool, WebTool,
};
use crate::agents::{AccessControlManager, AgentPool, TaskManager};
use crate::providers::Provider;
use crate::types::agent::PermissionMode;
use crate::types::session_task::SessionTaskList;
use crate::types::tool::{Tool, ToolContext, ToolResult, ToolUse};
use brainwires::permissions::{
    ActionOutcome, AuditEvent, AuditEventType, AuditLogger, PolicyAction, PolicyEngine,
    PolicyRequest, TrustLevel,
};

/// Extract domain from a URL string (simple parser without url crate dependency)
fn extract_domain_from_url(url: &str) -> Option<String> {
    // Remove protocol prefix
    let without_protocol = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);

    // Extract domain (everything before first '/' or end)
    let domain = without_protocol.split('/').next()?;

    // Remove port if present
    let domain = domain.split(':').next()?;

    // Remove userinfo if present (user:pass@domain)
    let domain = if domain.contains('@') {
        domain.split('@').next_back()?
    } else {
        domain
    };

    if domain.is_empty() {
        None
    } else {
        Some(domain.to_lowercase())
    }
}

/// Tool executor with approval management
pub struct ToolExecutor {
    registry: ToolRegistry,
    permission_mode: PermissionMode,
    /// Optional provider for agent-based tools (plan_task)
    provider: Option<Arc<dyn Provider>>,
    /// Orchestrator tool instance for programmatic tool calling
    orchestrator: OrchestratorTool,
    /// Whether orchestrator tool executors have been registered
    orchestrator_initialized: bool,
    /// Optional access control manager for inter-agent coordination
    access_control: Option<Arc<AccessControlManager>>,
    /// Optional agent pool for spawning background agents
    agent_pool: Option<Arc<RwLock<AgentPool>>>,
    /// Optional AgentPoolTool instance (created when agent_pool is set)
    agent_pool_tool: Option<AgentPoolTool>,
    /// Optional task manager for hierarchical task tracking
    task_manager: Option<Arc<RwLock<TaskManager>>>,
    /// Optional TaskManagerTool instance (created when task_manager is set)
    task_manager_tool: Option<TaskManagerTool>,
    /// Optional SessionTaskTool instance for session-specific task list
    session_task_tool: Option<SessionTaskTool>,
    /// Monitor tool — watch long-running background processes. Always on.
    monitor_tool: MonitorTool,
    /// Policy engine for declarative permission rules
    policy_engine: Option<PolicyEngine>,
    /// Audit logger for tracking tool executions
    audit_logger: Option<AuditLogger>,
    /// Approval channel for interactive approval requests
    approval_tx: Option<tokio::sync::mpsc::Sender<crate::approval::ApprovalRequest>>,
    /// Session-level approval decisions (tool_name -> response)
    session_approvals: std::collections::HashMap<String, crate::approval::ApprovalResponse>,
    /// Sudo password channel for interactive sudo password requests
    sudo_password_tx: Option<tokio::sync::mpsc::Sender<crate::sudo::SudoPasswordRequest>>,
    /// Remote bridge for permission relay (dangerous tool approval via web UI)
    remote_bridge: Option<
        std::sync::Arc<tokio::sync::RwLock<brainwires::agent_network::remote::RemoteBridge>>,
    >,
    /// Organization-level blocked tools (enforced client-side)
    org_blocked_tools: Vec<String>,
    /// Whether org policy forces permission relay for all dangerous tools
    org_permission_relay_required: bool,
    /// Whether org policy requires audit logging of all tool executions
    org_audit_all_commands: bool,
    /// Merged harness settings (permissions + hooks) from layered
    /// `settings.json` files. `None` = no layered settings configured;
    /// behavior falls back to the existing PolicyEngine / approval flow.
    settings: Option<std::sync::Arc<crate::config::Settings>>,
    /// Optional hook dispatcher for PreToolUse / PostToolUse / Stop events.
    hooks: Option<std::sync::Arc<crate::hooks::HookDispatcher>>,
    /// Optional channel for `ask_user_question` tool calls.
    user_question_tx: Option<tokio::sync::mpsc::Sender<crate::ask::UserQuestionRequest>>,
}

impl ToolExecutor {
    /// Create a new tool executor
    pub fn new(permission_mode: PermissionMode) -> Self {
        Self {
            registry: {
                let mut r = brainwires_tool_builtins::registry_with_builtins();
                r.register_tools(MonitorTool::get_tools());
                r.register_tools(MemoryTool::get_tools());
                r.register_tools(AskUserQuestionTool::get_tools());
                r
            },
            permission_mode,
            provider: None,
            orchestrator: OrchestratorTool::new(),
            orchestrator_initialized: false,
            access_control: None,
            agent_pool: None,
            agent_pool_tool: None,
            task_manager: None,
            task_manager_tool: None,
            session_task_tool: None,
            monitor_tool: MonitorTool::new(),
            policy_engine: None,
            audit_logger: None,
            approval_tx: None,
            session_approvals: std::collections::HashMap::new(),
            sudo_password_tx: None,
            remote_bridge: None,
            org_blocked_tools: Vec::new(),
            org_permission_relay_required: false,
            org_audit_all_commands: false,
            settings: None,
            hooks: None,
            user_question_tx: None,
        }
    }

    /// Create a new tool executor with a provider (enables agent-based tools)
    pub fn with_provider(permission_mode: PermissionMode, provider: Arc<dyn Provider>) -> Self {
        Self {
            registry: {
                let mut r = brainwires_tool_builtins::registry_with_builtins();
                r.register_tools(MonitorTool::get_tools());
                r.register_tools(MemoryTool::get_tools());
                r.register_tools(AskUserQuestionTool::get_tools());
                r
            },
            permission_mode,
            provider: Some(provider),
            orchestrator: OrchestratorTool::new(),
            orchestrator_initialized: false,
            access_control: None,
            agent_pool: None,
            agent_pool_tool: None,
            task_manager: None,
            task_manager_tool: None,
            session_task_tool: None,
            monitor_tool: MonitorTool::new(),
            policy_engine: None,
            audit_logger: None,
            approval_tx: None,
            session_approvals: std::collections::HashMap::new(),
            sudo_password_tx: None,
            remote_bridge: None,
            org_blocked_tools: Vec::new(),
            org_permission_relay_required: false,
            org_audit_all_commands: false,
            settings: None,
            hooks: None,
            user_question_tx: None,
        }
    }

    /// Set the remote bridge for permission relay (dangerous tool approval via web UI)
    pub fn set_remote_bridge(
        &mut self,
        bridge: std::sync::Arc<
            tokio::sync::RwLock<brainwires::agent_network::remote::RemoteBridge>,
        >,
    ) {
        self.remote_bridge = Some(bridge);
    }

    /// Apply organization-level policies received from the backend.
    ///
    /// These override local settings for blocked tools, forced permission relay,
    /// and audit logging. The server also enforces these server-side, but
    /// client-side enforcement provides faster UX feedback.
    pub fn apply_org_policies(
        &mut self,
        policies: &brainwires::agent_network::remote::OrgPolicies,
    ) {
        self.org_blocked_tools = policies.blocked_tools.clone();
        self.org_permission_relay_required = policies.permission_relay_required;
        self.org_audit_all_commands = policies.audit_all_commands;
    }

    /// Set the approval channel for interactive tool approval
    pub fn set_approval_channel(
        &mut self,
        tx: tokio::sync::mpsc::Sender<crate::approval::ApprovalRequest>,
    ) {
        self.approval_tx = Some(tx);
    }

    /// Attach layered `settings.json` permissions/hooks to this executor.
    /// Missing (`None`) behavior is identical to pre-settings code — existing
    /// PolicyEngine / approval paths still run.
    pub fn set_settings(&mut self, settings: std::sync::Arc<crate::config::Settings>) {
        self.settings = Some(settings);
    }

    /// Attach the hook dispatcher. Events fire at PreToolUse / PostToolUse.
    /// `UserPromptSubmit` and `Stop` are dispatched by the chat handler —
    /// this executor only fires the per-tool events.
    pub fn set_hooks(&mut self, hooks: std::sync::Arc<crate::hooks::HookDispatcher>) {
        self.hooks = Some(hooks);
    }

    /// Set the channel used by the `ask_user_question` tool.
    pub fn set_user_question_channel(
        &mut self,
        tx: tokio::sync::mpsc::Sender<crate::ask::UserQuestionRequest>,
    ) {
        self.user_question_tx = Some(tx);
    }

    /// Check if approval channel is configured
    pub fn has_approval_channel(&self) -> bool {
        self.approval_tx.is_some()
    }

    /// Set the sudo password channel for interactive sudo password requests
    pub fn set_sudo_password_channel(
        &mut self,
        tx: tokio::sync::mpsc::Sender<crate::sudo::SudoPasswordRequest>,
    ) {
        self.sudo_password_tx = Some(tx);
    }

    /// Check if a command string contains a sudo invocation
    fn command_needs_sudo(command: &str) -> bool {
        // Check each segment of piped/chained commands
        for segment in command.split(&['|', ';'][..]) {
            let trimmed = segment.trim();
            // Check for "&&" splits within the segment
            for part in trimmed.split("&&") {
                let part = part.trim();
                if part == "sudo" || part.starts_with("sudo ") {
                    return true;
                }
            }
        }
        false
    }

    /// Request sudo password from the user.
    /// In TUI mode, sends a request via the sudo channel.
    /// In CLI mode, falls back to dialoguer::Password via spawn_blocking.
    async fn request_sudo_password(&self, command: &str) -> Option<zeroize::Zeroizing<String>> {
        use crate::sudo::{SudoPasswordRequest, SudoPasswordResponse};

        if let Some(tx) = &self.sudo_password_tx {
            // TUI mode: send request via channel
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            let request = SudoPasswordRequest {
                id: uuid::Uuid::new_v4().to_string(),
                command: command.to_string(),
                response_tx,
            };
            if tx.send(request).await.is_err() {
                return None;
            }
            match response_rx.await {
                Ok(SudoPasswordResponse::Password(pw)) => Some(pw),
                _ => None,
            }
        } else {
            // CLI mode: use dialoguer::Password via spawn_blocking
            let cmd = command.to_string();
            tokio::task::spawn_blocking(move || {
                use dialoguer::Password;
                let username = std::env::var("USER")
                    .or_else(|_| std::env::var("LOGNAME"))
                    .unwrap_or_else(|_| "user".to_string());
                let cmd_display = if cmd.len() > 40 {
                    format!("{}...", &cmd[..40])
                } else {
                    cmd
                };
                let prompt = format!(
                    "[sudo] password for {} (command: {})",
                    username, cmd_display
                );
                match Password::new().with_prompt(&prompt).interact() {
                    Ok(pw) => Some(zeroize::Zeroizing::new(pw)),
                    Err(_) => None,
                }
            })
            .await
            .ok()
            .flatten()
        }
    }

    /// Record a session-level approval decision
    pub fn record_session_approval(
        &mut self,
        tool_name: &str,
        response: crate::approval::ApprovalResponse,
    ) {
        if response.is_session_persistent() {
            self.session_approvals
                .insert(tool_name.to_string(), response);
        }
    }

    /// Get session-level approval decision for a tool
    pub fn get_session_approval(
        &self,
        tool_name: &str,
    ) -> Option<crate::approval::ApprovalResponse> {
        self.session_approvals.get(tool_name).copied()
    }

    /// Determine if a tool requires user approval based on its category
    fn tool_requires_approval(&self, tool_name: &str) -> bool {
        matches!(
            tool_name,
            "write_file"
                | "edit_file"
                | "patch_file"
                | "delete_file"
                | "create_directory"
                | "execute_command"
                | "git_commit"
                | "git_push"
                | "git_reset"
                | "git_checkout"
                | "run_script"
                | "shell"
                | "bash"
        )
    }

    /// Create an ApprovalAction for a tool invocation
    fn create_approval_action(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> crate::approval::ApprovalAction {
        use crate::approval::ApprovalAction;

        match tool_name {
            "write_file" => {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown>")
                    .to_string();
                ApprovalAction::WriteFile { path }
            }
            "edit_file" | "patch_file" => {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown>")
                    .to_string();
                ApprovalAction::EditFile { path }
            }
            "delete_file" => {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown>")
                    .to_string();
                ApprovalAction::DeleteFile { path }
            }
            "create_directory" => {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown>")
                    .to_string();
                ApprovalAction::CreateDirectory { path }
            }
            "execute_command" | "run_script" | "shell" | "bash" => {
                let command = input
                    .get("command")
                    .and_then(|v| v.as_str())
                    .or_else(|| input.get("script").and_then(|v| v.as_str()))
                    .unwrap_or("<unknown>")
                    .to_string();
                ApprovalAction::ExecuteCommand { command }
            }
            "git_commit" | "git_push" | "git_reset" | "git_checkout" => ApprovalAction::GitModify {
                operation: tool_name
                    .strip_prefix("git_")
                    .unwrap_or(tool_name)
                    .to_string(),
            },
            _ => ApprovalAction::Other {
                description: format!("Execute tool: {}", tool_name),
            },
        }
    }

    /// Request approval from the user via remote bridge or local approval channel.
    ///
    /// If a remote bridge is active and has the PermissionRelay capability,
    /// the request is sent to the web UI. Otherwise, falls back to the local
    /// TUI approval dialog.
    async fn request_approval(
        &self,
        tool_name: &str,
        tool_description: &str,
        input: &serde_json::Value,
    ) -> Result<crate::approval::ApprovalResponse> {
        // Check session-level decisions first
        if let Some(decision) = self.session_approvals.get(tool_name) {
            return Ok(*decision);
        }

        // Try remote bridge first (if active and capable)
        if let Some(ref bridge_arc) = self.remote_bridge {
            let bridge = bridge_arc.read().await;
            if bridge.is_ready().await
                && bridge
                    .has_capability(
                        brainwires::agent_network::remote::ProtocolCapability::PermissionRelay,
                    )
                    .await
            {
                let _action = self.create_approval_action(tool_name, input);
                let action_desc = format!("{}: {}", tool_name, tool_description);

                match bridge
                    .send_permission_request("default", tool_name, &action_desc, input.clone())
                    .await
                {
                    Ok(decision) => {
                        if decision.approved {
                            if decision.always_allow {
                                return Ok(crate::approval::ApprovalResponse::ApproveForSession);
                            }
                            return Ok(crate::approval::ApprovalResponse::Approve);
                        } else {
                            if decision.remember_for_session {
                                return Ok(crate::approval::ApprovalResponse::DenyForSession);
                            }
                            return Ok(crate::approval::ApprovalResponse::Deny);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Remote permission relay failed, falling back to local: {}",
                            e
                        );
                        // Fall through to local approval
                    }
                }
            }
        }

        // Local TUI approval fallback
        let Some(ref tx) = self.approval_tx else {
            // No approval channel - default to approve (backward compatibility)
            return Ok(crate::approval::ApprovalResponse::Approve);
        };

        // Create the approval request
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let request = crate::approval::ApprovalRequest {
            id: uuid::Uuid::new_v4().to_string(),
            tool_name: tool_name.to_string(),
            action: self.create_approval_action(tool_name, input),
            details: crate::approval::ApprovalDetails {
                tool_description: tool_description.to_string(),
                parameters: input.clone(),
            },
            response_tx,
        };

        // Send the request to the TUI
        tx.send(request)
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;

        // Wait for the response
        let response = response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Approval response channel closed"))?;

        Ok(response)
    }

    /// Set the policy engine for declarative permission rules
    pub fn set_policy_engine(&mut self, engine: PolicyEngine) {
        self.policy_engine = Some(engine);
    }

    /// Set the policy engine with default security policies
    pub fn enable_policy_engine(&mut self) {
        self.policy_engine = Some(PolicyEngine::with_defaults());
    }

    /// Set the audit logger for tracking tool executions
    pub fn set_audit_logger(&mut self, logger: AuditLogger) {
        self.audit_logger = Some(logger);
    }

    /// Enable audit logging with default configuration
    pub fn enable_audit_logging(&mut self) -> Result<()> {
        self.audit_logger = Some(AuditLogger::new()?);
        Ok(())
    }

    /// Get reference to the policy engine
    pub fn policy_engine(&self) -> Option<&PolicyEngine> {
        self.policy_engine.as_ref()
    }

    /// Get reference to the audit logger
    pub fn audit_logger(&self) -> Option<&AuditLogger> {
        self.audit_logger.as_ref()
    }

    /// Set the provider for agent-based tools
    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = Some(provider);
    }

    /// Set the agent pool for spawning background agents
    pub fn set_agent_pool(&mut self, pool: Arc<RwLock<AgentPool>>) {
        self.agent_pool_tool = Some(AgentPoolTool::new(Arc::clone(&pool)));
        self.agent_pool = Some(pool);
    }

    /// Get reference to the agent pool
    pub fn agent_pool(&self) -> Option<&Arc<RwLock<AgentPool>>> {
        self.agent_pool.as_ref()
    }

    /// Set the task manager for hierarchical task tracking
    pub fn set_task_manager(&mut self, manager: Arc<RwLock<TaskManager>>) {
        self.task_manager_tool = Some(TaskManagerTool::new(Arc::clone(&manager)));
        self.task_manager = Some(manager);
    }

    /// Set the task manager with persistence support
    pub fn set_task_manager_with_persistence(
        &mut self,
        manager: Arc<RwLock<TaskManager>>,
        task_store: crate::storage::TaskStore,
        conversation_id: String,
    ) {
        self.task_manager_tool = Some(TaskManagerTool::with_persistence(
            Arc::clone(&manager),
            task_store,
            conversation_id,
        ));
        self.task_manager = Some(manager);
    }

    /// Get reference to the task manager
    pub fn task_manager(&self) -> Option<&Arc<RwLock<TaskManager>>> {
        self.task_manager.as_ref()
    }

    /// Set the session task list for session-specific task tracking
    pub fn set_session_task_list(&mut self, list: Arc<RwLock<SessionTaskList>>) {
        self.session_task_tool = Some(SessionTaskTool::new(list));
    }

    /// Set the access control manager for inter-agent coordination
    pub fn set_access_control(&mut self, acm: Arc<AccessControlManager>) {
        self.access_control = Some(acm);
    }

    /// Create access control manager with the given project root
    pub fn enable_access_control(&mut self, project_root: PathBuf) {
        self.access_control = Some(Arc::new(AccessControlManager::new(project_root)));
    }

    /// Get reference to the access control manager
    pub fn access_control(&self) -> Option<&Arc<AccessControlManager>> {
        self.access_control.as_ref()
    }

    /// Initialize the orchestrator with tool executors.
    ///
    /// This registers ALL tools from the registry as Rhai-callable functions.
    /// Tools are wrapped to be callable synchronously from Rhai scripts.
    ///
    /// Note: The permission system for individual tools within scripts will be
    /// handled by the orchestrator_permissions module (Phase 3).
    pub async fn initialize_orchestrator(&mut self) {
        if self.orchestrator_initialized {
            return;
        }

        // Register core file operations (synchronous, fast)
        self.register_file_ops_executors().await;

        // Register search executors
        self.register_search_executors().await;

        // Register git executors
        self.register_git_executors().await;

        // Register command executor (with safety restrictions)
        self.register_command_executor().await;

        // Register semantic search executors (RAG-based)
        self.register_semantic_search_executors().await;

        // Register context recall executor
        self.register_context_recall_executor().await;

        // Register code execution executor
        self.register_code_exec_executor().await;

        // Register MCP tools dynamically (from connected servers)
        self.register_mcp_executors().await;

        // Register agent pool executors if pool is configured
        self.register_agent_pool_executors().await;

        // Register task manager executors if task manager is configured
        self.register_task_manager_executors().await;

        // Register plan tool executor if provider is configured
        self.register_plan_executor().await;

        // Register fetch_url executor
        self.register_fetch_url_executor().await;

        // Register search_tools executor (meta-tool for discovering tools)
        self.register_search_tools_executor().await;

        self.orchestrator_initialized = true;
        tracing::info!("Orchestrator tool executors initialized with all tools including MCP");
    }

    /// Register file operation executors
    async fn register_file_ops_executors(&mut self) {
        // read_file
        self.orchestrator
            .register_executor("read_file", |input| {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "read_file requires 'path' parameter".to_string())?;
                std::fs::read_to_string(path)
                    .map_err(|e| format!("Failed to read file '{}': {}", path, e))
            })
            .await;

        // list_directory
        self.orchestrator
            .register_executor("list_directory", |input| {
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let entries: Vec<String> = std::fs::read_dir(path)
                    .map_err(|e| format!("Failed to list directory '{}': {}", path, e))?
                    .filter_map(|entry| {
                        entry
                            .ok()
                            .and_then(|e| e.path().to_str().map(|s| s.to_string()))
                    })
                    .collect();
                Ok(serde_json::to_string_pretty(&entries)
                    .unwrap_or_else(|_| format!("{:?}", entries)))
            })
            .await;

        // write_file
        self.orchestrator
            .register_executor("write_file", |input| {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "write_file requires 'path' parameter".to_string())?;
                let content = input
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "write_file requires 'content' parameter".to_string())?;
                std::fs::write(path, content)
                    .map_err(|e| format!("Failed to write file '{}': {}", path, e))?;
                // Read-back verification to detect concurrent clobber — see
                // FileOpsTool::write_file for the full rationale.
                let readback = std::fs::read(path)
                    .map_err(|e| format!("post-write readback failed for '{}': {}", path, e))?;
                if readback.as_slice() != content.as_bytes() {
                    return Err(format!(
                        "Write to {} succeeded but immediate read-back returned {} bytes \
                         (wrote {} bytes). This indicates concurrent modification by another \
                         process. Use a unique filename or coordinate with the other writer.",
                        path,
                        readback.len(),
                        content.len()
                    ));
                }
                Ok(format!(
                    "Successfully wrote {} bytes to {}",
                    content.len(),
                    path
                ))
            })
            .await;

        // edit_file (simple find/replace)
        self.orchestrator
            .register_executor("edit_file", |input| {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "edit_file requires 'path' parameter".to_string())?;
                let old_text = input
                    .get("old_text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "edit_file requires 'old_text' parameter".to_string())?;
                let new_text = input
                    .get("new_text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "edit_file requires 'new_text' parameter".to_string())?;

                let content = std::fs::read_to_string(path)
                    .map_err(|e| format!("Failed to read file '{}': {}", path, e))?;

                if !content.contains(old_text) {
                    return Err(format!("old_text not found in file '{}'", path));
                }

                let new_content = content.replacen(old_text, new_text, 1);
                std::fs::write(path, &new_content)
                    .map_err(|e| format!("Failed to write file '{}': {}", path, e))?;

                Ok(format!("Successfully edited {}", path))
            })
            .await;

        // create_directory
        self.orchestrator
            .register_executor("create_directory", |input| {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "create_directory requires 'path' parameter".to_string())?;
                std::fs::create_dir_all(path)
                    .map_err(|e| format!("Failed to create directory '{}': {}", path, e))?;
                Ok(format!("Successfully created directory {}", path))
            })
            .await;

        // delete_file
        self.orchestrator
            .register_executor("delete_file", |input| {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "delete_file requires 'path' parameter".to_string())?;
                std::fs::remove_file(path)
                    .map_err(|e| format!("Failed to delete file '{}': {}", path, e))?;
                Ok(format!("Successfully deleted {}", path))
            })
            .await;
    }

    /// Register search executors
    async fn register_search_executors(&mut self) {
        // search_files (find files matching pattern)
        self.orchestrator
            .register_executor("search_files", |input| {
                let pattern = input
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "search_files requires 'pattern' parameter".to_string())?;
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let output = std::process::Command::new("grep")
                    .args(["-r", "-l", pattern, path])
                    .output()
                    .map_err(|e| format!("Failed to search: {}", e))?;
                let files = String::from_utf8_lossy(&output.stdout);
                if files.is_empty() {
                    Ok("No files found matching pattern".to_string())
                } else {
                    Ok(files.to_string())
                }
            })
            .await;

        // search_code (search content with context)
        self.orchestrator
            .register_executor("search_code", |input| {
                let pattern = input
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "search_code requires 'pattern' parameter".to_string())?;
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let output = std::process::Command::new("rg")
                    .args(["--json", pattern, path])
                    .output()
                    .or_else(|_| {
                        std::process::Command::new("grep")
                            .args(["-r", "-n", pattern, path])
                            .output()
                    })
                    .map_err(|e| format!("Failed to search code: {}", e))?;
                let results = String::from_utf8_lossy(&output.stdout);
                if results.is_empty() {
                    Ok("No matches found".to_string())
                } else {
                    Ok(results.to_string())
                }
            })
            .await;
    }

    /// Register git executors
    async fn register_git_executors(&mut self) {
        // git_status
        self.orchestrator
            .register_executor("git_status", |_input| {
                let output = std::process::Command::new("git")
                    .args(["status", "--porcelain"])
                    .output()
                    .map_err(|e| format!("Failed to get git status: {}", e))?;
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            })
            .await;

        // git_diff
        self.orchestrator
            .register_executor("git_diff", |input| {
                let file = input.get("file").and_then(|v| v.as_str());
                let mut args = vec!["diff"];
                if let Some(f) = file {
                    args.push(f);
                }
                let output = std::process::Command::new("git")
                    .args(&args)
                    .output()
                    .map_err(|e| format!("Failed to get git diff: {}", e))?;
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            })
            .await;

        // git_log
        self.orchestrator
            .register_executor("git_log", |input| {
                let count = input.get("count").and_then(|v| v.as_i64()).unwrap_or(10);
                let output = std::process::Command::new("git")
                    .args(["log", "--oneline", &format!("-{}", count)])
                    .output()
                    .map_err(|e| format!("Failed to get git log: {}", e))?;
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            })
            .await;

        // git_show
        self.orchestrator
            .register_executor("git_show", |input| {
                let commit = input
                    .get("commit")
                    .and_then(|v| v.as_str())
                    .unwrap_or("HEAD");
                let output = std::process::Command::new("git")
                    .args(["show", "--stat", commit])
                    .output()
                    .map_err(|e| format!("Failed to get git show: {}", e))?;
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            })
            .await;
    }

    /// Register command executor with safety restrictions
    async fn register_command_executor(&mut self) {
        self.orchestrator
            .register_executor("execute_command", |input| {
                let command = input
                    .get("command")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "execute_command requires 'command' parameter".to_string())?;

                // Safety: only allow certain safe commands in orchestrator context
                let safe_prefixes = [
                    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "echo", "pwd", "date",
                    "which", "file", "stat",
                ];
                let is_safe = safe_prefixes
                    .iter()
                    .any(|prefix| command.starts_with(prefix));

                if !is_safe {
                    return Err(format!(
                        "Command '{}' not allowed in orchestrator context. Safe commands: {:?}",
                        command.split_whitespace().next().unwrap_or(""),
                        safe_prefixes
                    ));
                }

                let output = std::process::Command::new("sh")
                    .args(["-c", command])
                    .output()
                    .map_err(|e| format!("Failed to execute command: {}", e))?;

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    Ok(stdout.to_string())
                } else {
                    Err(format!("Command failed: {}", stderr))
                }
            })
            .await;
    }

    /// Register semantic search executors (RAG-based codebase search)
    async fn register_semantic_search_executors(&mut self) {
        // query_codebase - semantic search across indexed code
        self.orchestrator
            .register_executor("query_codebase", |input| {
                let query = input
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "query_codebase requires 'query' parameter".to_string())?;
                let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                let min_score = input
                    .get("min_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.7) as f32;

                // Execute async operation via runtime
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                rt.block_on(async {
                    SemanticSearchTool::execute_query(query, limit, min_score).await
                })
            })
            .await;

        // index_codebase - index a directory for semantic search
        self.orchestrator
            .register_executor("index_codebase", |input| {
                let path = input
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "index_codebase requires 'path' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                rt.block_on(async { SemanticSearchTool::execute_index(path).await })
            })
            .await;

        // search_with_filters - advanced search with filters
        self.orchestrator
            .register_executor("search_with_filters", |input| {
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                rt.block_on(async { SemanticSearchTool::execute_filtered_search(&input).await })
            })
            .await;

        // get_rag_statistics
        self.orchestrator
            .register_executor("get_rag_statistics", |_input| {
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                rt.block_on(async { SemanticSearchTool::execute_get_stats().await })
            })
            .await;

        // search_git_history
        self.orchestrator
            .register_executor("search_git_history", |input| {
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                rt.block_on(async { SemanticSearchTool::execute_git_history_search(&input).await })
            })
            .await;
    }

    /// Register context recall executor
    async fn register_context_recall_executor(&mut self) {
        self.orchestrator
            .register_executor("recall_context", |input| {
                let query = input
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "recall_context requires 'query' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                rt.block_on(async { ContextRecallTool::execute_recall(query).await })
            })
            .await;
    }

    /// Register code execution executor (Piston + Rhai)
    async fn register_code_exec_executor(&mut self) {
        self.orchestrator
            .register_executor("execute_code", |input| {
                let language = input
                    .get("language")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "execute_code requires 'language' parameter".to_string())?;
                let code = input
                    .get("code")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "execute_code requires 'code' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                rt.block_on(async { CodeExecTool::execute_code_helper(language, code).await })
            })
            .await;
    }

    /// Register MCP tool executors dynamically
    /// This registers all tools from connected MCP servers
    pub async fn register_mcp_executors(&mut self) {
        let mcp_tools = McpToolExecutor::get_all_tools().await;
        let tool_count = mcp_tools.len();

        for tool in mcp_tools {
            let tool_name = tool.name.clone();
            let tool_name_for_executor = tool_name.clone();
            self.orchestrator
                .register_executor(&tool_name, move |input| {
                    let rt = tokio::runtime::Handle::try_current()
                        .map_err(|_| "No tokio runtime available".to_string())?;

                    let name = tool_name_for_executor.clone();
                    rt.block_on(async move {
                        let registry = super::mcp_tool::MCP_TOOLS.read().await;
                        let arguments = if input.is_null() || input == serde_json::json!({}) {
                            None
                        } else {
                            Some(input)
                        };
                        registry
                            .execute_tool(&name, arguments)
                            .await
                            .map_err(|e| e.to_string())
                    })
                })
                .await;
        }

        tracing::info!("Registered {} MCP tools with orchestrator", tool_count);
    }

    /// Register agent pool executors for spawning/managing background agents
    async fn register_agent_pool_executors(&mut self) {
        // Only register if agent pool is configured
        let Some(pool) = &self.agent_pool else {
            tracing::debug!("Agent pool not configured, skipping agent pool executor registration");
            return;
        };

        let pool_clone = Arc::clone(pool);

        // agent_spawn - spawn a background task agent
        let pool_for_spawn = Arc::clone(&pool_clone);
        self.orchestrator
            .register_executor("agent_spawn", move |input| {
                let description = input
                    .get("description")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "agent_spawn requires 'description' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let pool = Arc::clone(&pool_for_spawn);
                let desc = description.to_string();

                rt.block_on(async move {
                    use crate::agents::TaskAgentConfig;
                    use crate::types::agent::{AgentContext, Task, TaskPriority};

                    let mut task = Task::new(uuid::Uuid::new_v4().to_string(), desc.clone());
                    task.set_priority(TaskPriority::Normal);

                    let config = TaskAgentConfig::default();
                    let context = AgentContext::default();

                    let pool_guard = pool.read().await;
                    pool_guard
                        .spawn_agent(task, context, Some(config))
                        .await
                        .map(|agent_id| {
                            format!("Spawned background agent '{}' for task: {}", agent_id, desc)
                        })
                        .map_err(|e| e.to_string())
                })
            })
            .await;

        // agent_status - get status of a background agent
        let pool_for_status = Arc::clone(&pool_clone);
        self.orchestrator
            .register_executor("agent_status", move |input| {
                let agent_id = input
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "agent_status requires 'agent_id' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let pool = Arc::clone(&pool_for_status);
                let id = agent_id.to_string();

                rt.block_on(async move {
                    let pool_guard = pool.read().await;
                    if let Some(status) = pool_guard.get_status(&id).await {
                        Ok(format!("Agent {}: {}", id, status))
                    } else {
                        Err(format!("Agent {} not found", id))
                    }
                })
            })
            .await;

        // agent_list - list all active agents
        let pool_for_list = Arc::clone(&pool_clone);
        self.orchestrator
            .register_executor("agent_list", move |_input| {
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let pool = Arc::clone(&pool_for_list);

                rt.block_on(async move {
                    let pool_guard = pool.read().await;
                    let agents = pool_guard.list_active().await;

                    if agents.is_empty() {
                        Ok("No active background agents".to_string())
                    } else {
                        let mut output = format!("{} active agents:\n", agents.len());
                        for (id, status) in agents {
                            output.push_str(&format!("- [{}] {}\n", id, status));
                        }
                        Ok(output)
                    }
                })
            })
            .await;

        // agent_stop - stop a running agent
        let pool_for_stop = Arc::clone(&pool_clone);
        self.orchestrator
            .register_executor("agent_stop", move |input| {
                let agent_id = input
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "agent_stop requires 'agent_id' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let pool = Arc::clone(&pool_for_stop);
                let id = agent_id.to_string();

                rt.block_on(async move {
                    let pool_guard = pool.read().await;
                    pool_guard
                        .stop_agent(&id)
                        .await
                        .map(|_| format!("Stopped agent {}", id))
                        .map_err(|e| e.to_string())
                })
            })
            .await;

        // agent_await - wait for agent completion
        let pool_for_await = Arc::clone(&pool_clone);
        self.orchestrator
            .register_executor("agent_await", move |input| {
                let agent_id = input
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "agent_await requires 'agent_id' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let pool = Arc::clone(&pool_for_await);
                let id = agent_id.to_string();

                rt.block_on(async move {
                    let pool_guard = pool.read().await;
                    pool_guard
                        .await_completion(&id)
                        .await
                        .map(|result| {
                            let status = if result.success {
                                "succeeded"
                            } else {
                                "failed"
                            };
                            format!(
                                "Agent {} {} after {} iterations:\n{}",
                                id, status, result.iterations, result.summary
                            )
                        })
                        .map_err(|e| e.to_string())
                })
            })
            .await;

        // agent_pool_stats - get pool statistics
        let pool_for_stats = Arc::clone(&pool_clone);
        self.orchestrator.register_executor("agent_pool_stats", move |_input| {
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| "No tokio runtime available".to_string())?;

            let pool = Arc::clone(&pool_for_stats);

            rt.block_on(async move {
                let pool_guard = pool.read().await;
                let stats = pool_guard.stats().await;

                Ok(format!(
                    "Agent Pool Statistics:\nMax agents: {}\nTotal agents: {}\nRunning: {}\nCompleted: {}\nFailed: {}",
                    stats.max_agents,
                    stats.total_agents,
                    stats.running,
                    stats.completed,
                    stats.failed
                ))
            })
        }).await;

        // agent_file_locks - list file locks held by agents
        let pool_for_locks = Arc::clone(&pool_clone);
        self.orchestrator
            .register_executor("agent_file_locks", move |_input| {
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let pool = Arc::clone(&pool_for_locks);

                rt.block_on(async move {
                    let pool_guard = pool.read().await;
                    let lock_manager = pool_guard.file_lock_manager();
                    let locks = lock_manager.list_locks().await;

                    if locks.is_empty() {
                        Ok("No file locks currently held".to_string())
                    } else {
                        let mut output = format!("{} file locks:\n", locks.len());
                        for (path, info) in locks {
                            let lock_type = match info.lock_type {
                                crate::agents::LockType::Read => "read",
                                crate::agents::LockType::Write => "write",
                            };
                            output.push_str(&format!(
                                "- {} ({}) by agent {}\n",
                                path.display(),
                                lock_type,
                                info.agent_id
                            ));
                        }
                        Ok(output)
                    }
                })
            })
            .await;

        tracing::info!("Registered 7 agent pool tools with orchestrator");
    }

    /// Register task manager executors for hierarchical task tracking
    async fn register_task_manager_executors(&mut self) {
        // Only register if task manager is configured
        let Some(manager) = &self.task_manager else {
            tracing::debug!(
                "Task manager not configured, skipping task manager executor registration"
            );
            return;
        };

        let manager_clone = Arc::clone(manager);

        // task_create - create a new task
        let manager_for_create = Arc::clone(&manager_clone);
        self.orchestrator
            .register_executor("task_create", move |input| {
                let description = input
                    .get("description")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "task_create requires 'description' parameter".to_string())?;
                let parent_id = input.get("parent_id").and_then(|v| v.as_str());

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let manager = Arc::clone(&manager_for_create);
                let desc = description.to_string();
                let parent = parent_id.map(|s| s.to_string());

                rt.block_on(async move {
                    use crate::types::agent::TaskPriority;
                    let mgr = manager.read().await;
                    let task_id = mgr
                        .create_task(desc, parent, TaskPriority::Normal)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok(format!("Created task: {}", task_id))
                })
            })
            .await;

        // task_start - start a task
        let manager_for_start = Arc::clone(&manager_clone);
        self.orchestrator
            .register_executor("task_start", move |input| {
                let task_id = input
                    .get("task_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "task_start requires 'task_id' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let manager = Arc::clone(&manager_for_start);
                let id = task_id.to_string();

                rt.block_on(async move {
                    let mgr = manager.read().await;
                    mgr.start_task(&id).await.map_err(|e| e.to_string())?;
                    Ok(format!("Started task: {}", id))
                })
            })
            .await;

        // task_complete - complete a task
        let manager_for_complete = Arc::clone(&manager_clone);
        self.orchestrator
            .register_executor("task_complete", move |input| {
                let task_id = input
                    .get("task_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "task_complete requires 'task_id' parameter".to_string())?;
                let result = input.get("result").and_then(|v| v.as_str()).unwrap_or("");

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let manager = Arc::clone(&manager_for_complete);
                let id = task_id.to_string();
                let res = result.to_string();

                rt.block_on(async move {
                    let mgr = manager.read().await;
                    mgr.complete_task(&id, res)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok(format!("Completed task: {}", id))
                })
            })
            .await;

        // task_fail - fail a task
        let manager_for_fail = Arc::clone(&manager_clone);
        self.orchestrator
            .register_executor("task_fail", move |input| {
                let task_id = input
                    .get("task_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "task_fail requires 'task_id' parameter".to_string())?;
                let reason = input
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let manager = Arc::clone(&manager_for_fail);
                let id = task_id.to_string();
                let rsn = reason.to_string();

                rt.block_on(async move {
                    let mgr = manager.read().await;
                    mgr.fail_task(&id, rsn).await.map_err(|e| e.to_string())?;
                    Ok(format!("Failed task: {}", id))
                })
            })
            .await;

        // task_tree - get the task tree
        let manager_for_tree = Arc::clone(&manager_clone);
        self.orchestrator
            .register_executor("task_tree", move |_input| {
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let manager = Arc::clone(&manager_for_tree);

                rt.block_on(async move {
                    let mgr = manager.read().await;
                    Ok(mgr.format_tree().await)
                })
            })
            .await;

        // task_ready - get ready tasks
        let manager_for_ready = Arc::clone(&manager_clone);
        self.orchestrator
            .register_executor("task_ready", move |_input| {
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let manager = Arc::clone(&manager_for_ready);

                rt.block_on(async move {
                    let mgr = manager.read().await;
                    let ready = mgr.get_ready_tasks().await;
                    if ready.is_empty() {
                        Ok("No ready tasks".to_string())
                    } else {
                        let mut output = format!("{} ready tasks:\n", ready.len());
                        for task in ready {
                            output.push_str(&format!("- [{}] {}\n", task.id, task.description));
                        }
                        Ok(output)
                    }
                })
            })
            .await;

        // task_stats - get task statistics
        let manager_for_stats = Arc::clone(&manager_clone);
        self.orchestrator.register_executor("task_stats", move |_input| {
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| "No tokio runtime available".to_string())?;

            let manager = Arc::clone(&manager_for_stats);

            rt.block_on(async move {
                let mgr = manager.read().await;
                let stats = mgr.get_stats().await;
                Ok(format!(
                    "Task Statistics:\nTotal: {}\nPending: {}\nIn Progress: {}\nCompleted: {}\nFailed: {}",
                    stats.total, stats.pending, stats.in_progress, stats.completed, stats.failed
                ))
            })
        }).await;

        tracing::info!("Registered 7 task manager tools with orchestrator");
    }

    /// Register plan tool executor if provider is configured
    async fn register_plan_executor(&mut self) {
        // Only register if provider is configured
        let Some(provider) = &self.provider else {
            tracing::debug!("Provider not configured, skipping plan tool executor registration");
            return;
        };

        let provider_clone = Arc::clone(provider);

        self.orchestrator
            .register_executor("plan_task", move |input| {
                let task = input
                    .get("task")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "plan_task requires 'task' parameter".to_string())?;
                let context = input.get("context").and_then(|v| v.as_str()).unwrap_or("");
                let max_iterations = input
                    .get("max_iterations")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as u32;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let provider = Arc::clone(&provider_clone);
                let task_str = task.to_string();
                let context_str = context.to_string();

                rt.block_on(async move {
                    let plan_tool = PlanTool::new(provider);
                    let input = serde_json::json!({
                        "task": task_str,
                        "context": context_str,
                        "max_iterations": max_iterations
                    });
                    let result = plan_tool.execute("orchestrator", "plan_task", &input).await;
                    if result.is_error {
                        Err(result.content)
                    } else {
                        Ok(result.content)
                    }
                })
            })
            .await;

        tracing::info!("Registered plan_task tool with orchestrator");
    }

    /// Register fetch_url executor for fetching web content
    async fn register_fetch_url_executor(&mut self) {
        self.orchestrator
            .register_executor("fetch_url", |input| {
                let url = input
                    .get("url")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "fetch_url requires 'url' parameter".to_string())?;

                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| "No tokio runtime available".to_string())?;

                let url_str = url.to_string();

                rt.block_on(async move {
                    WebTool::fetch_url_content(&url_str)
                        .await
                        .map_err(|e| e.to_string())
                })
            })
            .await;

        tracing::info!("Registered fetch_url tool with orchestrator");
    }

    /// Register search_tools executor (meta-tool for discovering tools)
    async fn register_search_tools_executor(&mut self) {
        // Note: search_tools needs access to the registry, but we can't easily
        // pass it to the closure. Instead, we'll make it return a helpful message
        // directing users to use the tool directly via normal tool calling.
        self.orchestrator.register_executor("search_tools", |input| {
            let query = input.get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "search_tools requires 'query' parameter".to_string())?;

            // Since we can't access the registry from within the Rhai script context,
            // return a message about available tool categories
            Ok(format!(
                "Tool search for '{}' - Available tool categories in orchestrator:\n\
                \n\
                File Operations: read_file, write_file, edit_file, list_directory, create_directory, delete_file\n\
                Search: search_files, search_code\n\
                Git: git_status, git_diff, git_log, git_show\n\
                Commands: execute_command (safe commands only)\n\
                Semantic Search: query_codebase, index_codebase, search_with_filters, get_rag_statistics, search_git_history\n\
                Web: fetch_url\n\
                Context: recall_context\n\
                Code Execution: execute_code\n\
                Agent Pool: agent_spawn, agent_status, agent_list, agent_stop, agent_await, agent_pool_stats, agent_file_locks\n\
                Task Manager: task_create, task_start, task_complete, task_fail, task_tree, task_ready, task_stats\n\
                Planning: plan_task\n\
                MCP: All connected MCP server tools (mcp_* prefix)\n\
                \n\
                For detailed tool schemas, use the search_tools tool via direct API call.",
                query
            ))
        }).await;

        tracing::info!("Registered search_tools tool with orchestrator");
    }

    /// Get the orchestrator for programmatic tool calling
    pub fn orchestrator(&self) -> &OrchestratorTool {
        &self.orchestrator
    }

    /// Get all registered tools
    pub fn get_tools(&self) -> &[Tool] {
        self.registry.get_all()
    }

    /// Get reference to the tool registry
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Execute a tool use request
    pub async fn execute(&self, tool_use: &ToolUse, context: &ToolContext) -> Result<ToolResult> {
        // Org audit logging: log every tool execution when org policy requires it
        if self.org_audit_all_commands {
            tracing::info!(target: "org_audit", tool = %tool_use.name, "org audit: tool execution");
        }

        // Find the tool
        let tool = self
            .registry
            .get(&tool_use.name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", tool_use.name))?;

        // Org policy: blocked tools check (before any capability/policy checks)
        if self.org_blocked_tools.iter().any(|t| t == &tool_use.name) {
            return Ok(ToolResult::error(
                tool_use.id.clone(),
                format!("Tool '{}' is blocked by organization policy", tool_use.name),
            ));
        }

        // Layered-settings permission check (deny > ask > allow). Runs
        // before the PolicyEngine branch so `deny` trumps everything,
        // including `PermissionMode::Full`.
        let settings_decision = self
            .settings
            .as_ref()
            .and_then(|s| s.permissions.as_ref())
            .map(|p| p.decide(&tool_use.name, &tool_use.input))
            .unwrap_or(crate::config::PermissionDecision::Unset);

        if let crate::config::PermissionDecision::Deny(ref reason) = settings_decision {
            tracing::warn!(
                "Tool '{}' denied by layered settings: {}",
                tool_use.name,
                reason
            );
            if let Some(ref audit_logger) = self.audit_logger {
                let agent_id = context
                    .metadata
                    .get("agent_id")
                    .cloned()
                    .unwrap_or_else(|| "main".to_string());
                let _ = audit_logger.log_denied(
                    Some(&agent_id),
                    &format!("tool:{}", tool_use.name),
                    tool_use.input.get("path").and_then(|v| v.as_str()),
                    reason,
                );
            }
            return Ok(ToolResult::error(
                tool_use.id.clone(),
                format!("Tool '{}' {}", tool_use.name, reason),
            ));
        }

        // Check capabilities if present (new capability-based permission system)
        // Deserialize from JSON value to concrete AgentCapabilities type
        let capabilities: Option<brainwires::permissions::AgentCapabilities> = context
            .capabilities
            .as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        if let Some(ref capabilities) = capabilities {
            // Check if tool is allowed by capabilities
            if !capabilities.allows_tool(&tool_use.name) {
                tracing::warn!(
                    "Tool '{}' denied by capability system (category: {:?})",
                    tool_use.name,
                    brainwires::permissions::AgentCapabilities::categorize_tool(&tool_use.name)
                );
                return Ok(ToolResult::error(
                    tool_use.id.clone(),
                    format!(
                        "Tool '{}' is not permitted by agent capabilities",
                        tool_use.name
                    ),
                ));
            }

            // Check if tool requires explicit approval regardless of trust
            // (Actual approval logic is handled later in the function via approval channel)
            if capabilities.requires_approval(&tool_use.name)
                && self.permission_mode == PermissionMode::Auto
            {
                tracing::debug!(
                    "Tool '{}' flagged for approval (capability policy) - will be checked via approval channel",
                    tool_use.name
                );
            }

            // For file operations, check path permissions
            if let Some(path) = tool_use.input.get("path").and_then(|v| v.as_str()) {
                let is_write = matches!(
                    tool_use.name.as_str(),
                    "write_file" | "edit_file" | "patch_file" | "delete_file" | "create_directory"
                );

                if is_write && !capabilities.allows_write(path) {
                    tracing::warn!("Write to path '{}' denied by capability system", path);
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!(
                            "Write access to '{}' is not permitted by agent capabilities",
                            path
                        ),
                    ));
                }

                if !is_write && !capabilities.allows_read(path) {
                    tracing::warn!("Read from path '{}' denied by capability system", path);
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!(
                            "Read access to '{}' is not permitted by agent capabilities",
                            path
                        ),
                    ));
                }
            }

            // For network operations, check domain permissions
            if let Some(url_str) = tool_use.input.get("url").and_then(|v| v.as_str()) {
                // Simple domain extraction from URL
                if let Some(domain) = extract_domain_from_url(url_str)
                    && !capabilities.allows_domain(&domain)
                {
                    tracing::warn!(
                        "Network access to domain '{}' denied by capability system",
                        domain
                    );
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!(
                            "Network access to '{}' is not permitted by agent capabilities",
                            domain
                        ),
                    ));
                }
            }
        }

        // Get agent ID from context metadata (defaults to "main" for direct execution)
        let agent_id = context
            .metadata
            .get("agent_id")
            .cloned()
            .unwrap_or_else(|| "main".to_string());

        // Policy engine evaluation (declarative permission rules)
        if let Some(ref policy_engine) = self.policy_engine {
            // Build policy request from tool invocation context
            let file_path = tool_use
                .input
                .get("path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let domain = tool_use
                .input
                .get("url")
                .and_then(|v| v.as_str())
                .and_then(extract_domain_from_url);
            let git_op = if tool_use.name.starts_with("git_") {
                brainwires::permissions::config::parse_git_operation(&tool_use.name)
            } else {
                None
            };

            // Get trust level from context or default to Medium (2)
            let trust_level = context
                .metadata
                .get("trust_level")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(TrustLevel::Medium.as_u8());

            let request = PolicyRequest {
                tool_name: Some(tool_use.name.clone()),
                tool_category: Some(brainwires::permissions::AgentCapabilities::categorize_tool(
                    &tool_use.name,
                )),
                file_path,
                domain,
                git_operation: git_op,
                trust_level,
                agent_id: Some(agent_id.clone()),
                metadata: std::collections::HashMap::new(),
            };

            let decision = policy_engine.evaluate(&request);

            match decision.action {
                PolicyAction::Deny => {
                    tracing::warn!(
                        "Tool '{}' denied by policy '{}'",
                        tool_use.name,
                        decision.matched_policy.as_deref().unwrap_or("default")
                    );
                    // Log denial to audit if enabled
                    if let Some(ref audit_logger) = self.audit_logger {
                        let _ = audit_logger.log_denied(
                            Some(&agent_id),
                            &format!("tool:{}", tool_use.name),
                            tool_use.input.get("path").and_then(|v| v.as_str()),
                            decision
                                .matched_policy
                                .as_deref()
                                .unwrap_or("policy_denied"),
                        );
                    }
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!("Tool '{}' denied by policy", tool_use.name),
                    ));
                }
                PolicyAction::DenyWithMessage(ref msg) => {
                    tracing::warn!(
                        "Tool '{}' denied by policy '{}': {}",
                        tool_use.name,
                        decision.matched_policy.as_deref().unwrap_or("default"),
                        msg
                    );
                    if let Some(ref audit_logger) = self.audit_logger {
                        let _ = audit_logger.log_denied(
                            Some(&agent_id),
                            &format!("tool:{}", tool_use.name),
                            tool_use.input.get("path").and_then(|v| v.as_str()),
                            msg,
                        );
                    }
                    return Ok(ToolResult::error(tool_use.id.clone(), msg.clone()));
                }
                PolicyAction::RequireApproval => {
                    tracing::info!(
                        "Tool '{}' requires approval per policy '{}' - will be checked via approval channel",
                        tool_use.name,
                        decision.matched_policy.as_deref().unwrap_or("default")
                    );
                    // Approval is handled later in the function via the approval channel
                    // For non-auto mode, deny immediately since we can't prompt
                    if self.permission_mode != PermissionMode::Auto {
                        if let Some(ref audit_logger) = self.audit_logger {
                            let _ = audit_logger.log_denied(
                                Some(&agent_id),
                                &format!("tool:{}", tool_use.name),
                                tool_use.input.get("path").and_then(|v| v.as_str()),
                                "approval_required_but_not_auto_mode",
                            );
                        }
                        return Ok(ToolResult::error(
                            tool_use.id.clone(),
                            format!("Tool '{}' requires approval", tool_use.name),
                        ));
                    }
                }
                PolicyAction::AllowWithAudit => {
                    tracing::debug!(
                        "Tool '{}' allowed with audit per policy '{}'",
                        tool_use.name,
                        decision.matched_policy.as_deref().unwrap_or("default")
                    );
                    // Audit logging will happen after execution
                }
                PolicyAction::Escalate => {
                    tracing::warn!(
                        "Tool '{}' escalated by policy '{}' - treating as RequireApproval",
                        tool_use.name,
                        decision.matched_policy.as_deref().unwrap_or("default")
                    );
                    if self.permission_mode != PermissionMode::Auto {
                        return Ok(ToolResult::error(
                            tool_use.id.clone(),
                            format!(
                                "Tool '{}' escalated - requires higher authorization",
                                tool_use.name
                            ),
                        ));
                    }
                }
                PolicyAction::Allow => {
                    // Allowed, continue execution
                }
            }
        }

        // Legacy permission mode checks (for backward compatibility)
        if tool.requires_approval && self.permission_mode == PermissionMode::ReadOnly {
            return Ok(ToolResult::error(
                tool_use.id.clone(),
                format!(
                    "Tool '{}' requires approval but permission mode is read-only",
                    tool_use.name
                ),
            ));
        }

        // In auto mode, check if approval is needed and request it via the approval channel.
        // Org policy: when permission_relay_required is set, force approval for any tool
        // that matches tool_requires_approval(), regardless of the tool's own flag.
        let mut needs_approval = (tool.requires_approval
            || self.tool_requires_approval(&tool_use.name))
            && self.permission_mode == PermissionMode::Auto
            && self.approval_tx.is_some();
        if self.org_permission_relay_required && self.tool_requires_approval(&tool_use.name) {
            needs_approval = true;
        }
        // Layered settings override: `allow` skips approval; `ask` forces it
        // even on tools that normally wouldn't prompt. `deny` already
        // short-circuited above.
        match settings_decision {
            crate::config::PermissionDecision::Allow => {
                needs_approval = false;
            }
            crate::config::PermissionDecision::Ask => {
                if self.approval_tx.is_some() {
                    needs_approval = true;
                }
            }
            _ => {}
        }

        if needs_approval {
            tracing::info!(
                "Tool '{}' requires approval - requesting from user",
                tool_use.name
            );

            match self
                .request_approval(&tool_use.name, &tool.description, &tool_use.input)
                .await
            {
                Ok(response) => {
                    use crate::approval::ApprovalResponse;
                    match response {
                        ApprovalResponse::Approve | ApprovalResponse::ApproveForSession => {
                            tracing::info!("Tool '{}' approved by user", tool_use.name);
                            // Continue with execution
                        }
                        ApprovalResponse::Deny | ApprovalResponse::DenyForSession => {
                            tracing::info!("Tool '{}' denied by user", tool_use.name);
                            return Ok(ToolResult::error(
                                tool_use.id.clone(),
                                format!("Tool '{}' was denied by user", tool_use.name),
                            ));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Approval request failed for tool '{}': {}",
                        tool_use.name,
                        e
                    );
                    // If approval channel is broken, deny by default for safety
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!("Tool '{}' approval failed: {}", tool_use.name, e),
                    ));
                }
            }
        }

        // Acquire locks if access control is enabled
        let _lock_bundle = if let Some(acm) = &self.access_control {
            match acm
                .acquire_for_tool(&agent_id, &tool_use.name, &tool_use.input)
                .await
            {
                Ok(bundle) => Some(bundle),
                Err(e) => {
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!("Access control error: {}", e),
                    ));
                }
            }
        } else {
            None
        };

        // PreToolUse hook — fires after approval, before dispatch. Exit 2
        // blocks the tool; other failures are logged but don't stop execution.
        if let Some(ref hooks) = self.hooks {
            match hooks
                .dispatch_pre_tool(&tool_use.name, &tool_use.input)
                .await
            {
                crate::hooks::HookOutcome::Continue => {}
                crate::hooks::HookOutcome::Block { reason } => {
                    tracing::info!(
                        "Tool '{}' blocked by PreToolUse hook: {}",
                        tool_use.name,
                        reason
                    );
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!("Blocked by PreToolUse hook: {}", reason),
                    ));
                }
                crate::hooks::HookOutcome::SoftError(msg) => {
                    tracing::warn!("PreToolUse hook error (continuing): {}", msg);
                }
            }
        }

        // Route to the appropriate tool implementation
        tracing::debug!("[ToolExecutor] Routing tool: {}", tool_use.name);
        let result = self
            .route_tool(&tool_use.id, &tool_use.name, &tool_use.input, context)
            .await;
        tracing::debug!(
            "[ToolExecutor] Tool {} completed: is_error={}",
            tool_use.name,
            result.is_error
        );

        // PostToolUse hook — observes the tool result. Block outcome here
        // does NOT undo the side-effects (the tool has already run) but is
        // surfaced as an error so the model sees the feedback.
        if let Some(ref hooks) = self.hooks {
            let result_payload = serde_json::json!({
                "content": result.content,
            });
            match hooks
                .dispatch_post_tool(
                    &tool_use.name,
                    &tool_use.input,
                    &result_payload,
                    result.is_error,
                )
                .await
            {
                crate::hooks::HookOutcome::Continue => {}
                crate::hooks::HookOutcome::Block { reason } => {
                    tracing::info!(
                        "PostToolUse hook signalled block for '{}': {}",
                        tool_use.name,
                        reason
                    );
                    return Ok(ToolResult::error(
                        tool_use.id.clone(),
                        format!("PostToolUse hook: {}", reason),
                    ));
                }
                crate::hooks::HookOutcome::SoftError(msg) => {
                    tracing::warn!("PostToolUse hook error (continuing): {}", msg);
                }
            }
        }

        // Track successful file reads for read-before-write enforcement
        if let Some(acm) = &self.access_control
            && tool_use.name == "read_file"
            && !result.is_error
            && let Some(path_str) = tool_use.input.get("path").and_then(|v| v.as_str())
        {
            let path = if std::path::Path::new(path_str).is_absolute() {
                PathBuf::from(path_str)
            } else {
                PathBuf::from(&context.working_directory).join(path_str)
            };
            acm.track_file_read(&agent_id, &path).await;
        }

        // Audit log the tool execution
        if let Some(ref audit_logger) = self.audit_logger {
            let target = tool_use
                .input
                .get("path")
                .and_then(|v| v.as_str())
                .or_else(|| tool_use.input.get("url").and_then(|v| v.as_str()))
                .unwrap_or(&tool_use.name);

            let outcome = if result.is_error {
                ActionOutcome::Failure
            } else {
                ActionOutcome::Success
            };

            let trust_level = context
                .metadata
                .get("trust_level")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(TrustLevel::Medium.as_u8());

            let event = AuditEvent::new(AuditEventType::ToolExecution)
                .with_agent(&agent_id)
                .with_action(&format!("tool:{}", tool_use.name))
                .with_target(target)
                .with_trust_level(trust_level)
                .with_outcome(outcome);

            // Best-effort logging - don't fail the tool execution if audit fails
            if let Err(e) = audit_logger.log(event) {
                tracing::warn!("Failed to log audit event: {}", e);
            }
        }

        // Lock bundle automatically released when dropped
        Ok(result)
    }

    /// Execute a tool with automatic retry for transient errors
    ///
    /// Uses error classification from the error taxonomy to determine:
    /// 1. Whether the error is retryable (transient, external service)
    /// 2. What retry strategy to use (exponential backoff, fixed delay)
    /// 3. When to give up and return the error
    ///
    /// Returns a `ToolOutcome` for SEAL learning integration.
    pub async fn execute_with_retry(
        &self,
        tool_use: &ToolUse,
        context: &ToolContext,
    ) -> Result<(ToolResult, super::error::ToolOutcome)> {
        use super::error::{ToolOutcome, classify_error};
        use std::time::Instant;

        let start_time = Instant::now();
        let mut attempt = 0u32;
        let mut last_result: Option<ToolResult> = None;

        loop {
            let result = self.execute(tool_use, context).await?;

            if !result.is_error {
                // Success - return with outcome for SEAL
                let outcome = ToolOutcome::success(
                    &tool_use.name,
                    attempt,
                    start_time.elapsed().as_millis() as u64,
                );
                return Ok((result, outcome));
            }

            // Classify the error to determine retry strategy
            let error_category = classify_error(&tool_use.name, &result.content);

            if !error_category.is_retryable() {
                // Non-retryable error - fail immediately
                tracing::debug!(
                    tool = %tool_use.name,
                    error_type = %error_category.category_name(),
                    "Tool error not retryable"
                );
                let outcome = ToolOutcome::failure(
                    &tool_use.name,
                    attempt,
                    error_category,
                    start_time.elapsed().as_millis() as u64,
                );
                return Ok((result, outcome));
            }

            // Get retry strategy
            let strategy = error_category.retry_strategy();

            // Check if we should retry
            if let Some(delay) = strategy.delay_for_attempt(attempt) {
                attempt += 1;
                tracing::info!(
                    tool = %tool_use.name,
                    attempt = attempt,
                    max_attempts = strategy.max_attempts(),
                    delay_ms = delay.as_millis() as u64,
                    error_type = %error_category.category_name(),
                    "Retrying tool after transient error"
                );

                // Wait before retry
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }

                last_result = Some(result);
                continue;
            }

            // Max retries exceeded
            tracing::warn!(
                tool = %tool_use.name,
                attempts = attempt + 1,
                error_type = %error_category.category_name(),
                "Tool failed after max retries"
            );
            let outcome = ToolOutcome::failure(
                &tool_use.name,
                attempt,
                error_category,
                start_time.elapsed().as_millis() as u64,
            );
            return Ok((last_result.unwrap_or(result), outcome));
        }
    }

    /// Route tool execution to the appropriate implementation
    async fn route_tool(
        &self,
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        tracing::debug!("[route_tool] Routing: {}", tool_name);
        // Determine which category the tool belongs to
        if tool_name.starts_with("read_file")
            || tool_name.starts_with("write_file")
            || tool_name.starts_with("edit_file")
            || tool_name.starts_with("patch_file")
            || tool_name.starts_with("list_directory")
            || tool_name.starts_with("search_files")
            || tool_name.starts_with("delete_file")
            || tool_name.starts_with("create_directory")
        {
            tracing::debug!("[route_tool] Executing FileOpsTool::{}", tool_name);
            let result = FileOpsTool::execute(tool_use_id, tool_name, input, context);
            tracing::debug!("[route_tool] FileOpsTool::{} returned", tool_name);
            result
        } else if tool_name.starts_with("execute_command") {
            // Check if the command needs sudo — if so, request password and use sudo execution
            let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if Self::command_needs_sudo(command) {
                match self.request_sudo_password(command).await {
                    Some(password) => BashTool::execute_with_sudo(
                        tool_use_id,
                        tool_name,
                        input,
                        context,
                        password,
                    ),
                    None => ToolResult::error(
                        tool_use_id.to_string(),
                        "Sudo command cancelled: password not provided.".to_string(),
                    ),
                }
            } else {
                BashTool::execute(tool_use_id, tool_name, input, context)
            }
        } else if tool_name.starts_with("git_") {
            GitTool::execute(tool_use_id, tool_name, input, context)
        } else if tool_name.starts_with("fetch_url") {
            WebTool::execute(tool_use_id, tool_name, input, context).await
        } else if tool_name.starts_with("search_code") {
            SearchTool::execute(tool_use_id, tool_name, input, context)
        } else if tool_name == "index_codebase"
            || tool_name == "query_codebase"
            || tool_name == "search_with_filters"
            || tool_name == "get_rag_statistics"
            || tool_name == "clear_rag_index"
            || tool_name == "search_git_history"
        {
            SemanticSearchTool::execute(tool_use_id, tool_name, input, context).await
        } else if tool_name.starts_with("mcp_") {
            McpToolExecutor::execute(tool_use_id, tool_name, input, context)
        } else if tool_name == "plan_task" {
            // Plan tool requires a provider
            if let Some(provider) = &self.provider {
                let plan_tool = PlanTool::new(Arc::clone(provider));
                plan_tool.execute(tool_use_id, tool_name, input).await
            } else {
                ToolResult::error(
                    tool_use_id.to_string(),
                    "plan_task requires a provider to be configured. Use ToolExecutor::with_provider() or set_provider().".to_string()
                )
            }
        } else if tool_name == "recall_context" {
            // Context recall tool for searching conversation history
            ContextRecallTool::execute(tool_use_id, tool_name, input, context).await
        } else if tool_name == "search_tools" {
            // Tool search meta-tool for dynamic tool discovery
            ToolSearchTool::execute(tool_use_id, tool_name, input, context, &self.registry)
        } else if tool_name == "execute_script" {
            // Orchestrator tool for programmatic tool calling
            self.orchestrator
                .execute(tool_use_id, tool_name, input, context)
                .await
        } else if tool_name == "execute_code" {
            // Code execution tool (Piston + Rhai)
            CodeExecTool::execute(tool_use_id, tool_name, input, context).await
        } else if tool_name.starts_with("agent_") {
            // Agent pool tools for spawning/managing background agents
            if let Some(agent_pool_tool) = &self.agent_pool_tool {
                agent_pool_tool.execute(tool_use_id, tool_name, input).await
            } else {
                ToolResult::error(
                    tool_use_id.to_string(),
                    "Agent pool tools require an agent pool to be configured. Use ToolExecutor::set_agent_pool().".to_string()
                )
            }
        } else if tool_name.starts_with("monitor_") {
            // Background process watcher (long-running dev servers, log tails, etc.)
            self.monitor_tool
                .execute(tool_use_id, tool_name, input)
                .await
        } else if tool_name.starts_with("memory_") {
            // Per-project auto-memory (MEMORY.md + typed memory files).
            let cwd = std::path::PathBuf::from(&context.working_directory);
            MemoryTool::new()
                .execute(tool_use_id, tool_name, input, &cwd)
                .await
        } else if tool_name == "ask_user_question" {
            // Ask the user directly — channel when TUI is active, dialoguer
            // otherwise, Cancelled on non-TTY.
            AskUserQuestionTool::new(self.user_question_tx.clone())
                .execute(tool_use_id, input)
                .await
        } else if tool_name == "task_list_write" {
            // Session task tool for session-specific task list tracking
            if let Some(session_task_tool) = &self.session_task_tool {
                session_task_tool
                    .execute(tool_use_id, tool_name, input)
                    .await
            } else {
                ToolResult::error(
                    tool_use_id.to_string(),
                    "Session task tool requires a session task list to be configured. Use ToolExecutor::set_session_task_list().".to_string()
                )
            }
        } else if tool_name.starts_with("task_") {
            // Task manager tools for hierarchical task tracking
            if let Some(task_manager_tool) = &self.task_manager_tool {
                task_manager_tool
                    .execute(tool_use_id, tool_name, input)
                    .await
            } else {
                ToolResult::error(
                    tool_use_id.to_string(),
                    "Task manager tools require a task manager to be configured. Use ToolExecutor::set_task_manager().".to_string()
                )
            }
        } else if tool_name == "check_duplicates" {
            // Validation tools for code quality
            let file_path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            match crate::tools::validation_tools::check_duplicates(file_path).await {
                Ok(mut result) => {
                    result.tool_use_id = tool_use_id.to_string();
                    result
                }
                Err(e) => ToolResult::error(tool_use_id.to_string(), e.to_string()),
            }
        } else if tool_name == "verify_build" {
            let working_directory = input
                .get("working_directory")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            let build_type = input
                .get("build_type")
                .and_then(|v| v.as_str())
                .unwrap_or("npm");
            match crate::tools::validation_tools::verify_build(working_directory, build_type).await
            {
                Ok(mut result) => {
                    result.tool_use_id = tool_use_id.to_string();
                    result
                }
                Err(e) => ToolResult::error(tool_use_id.to_string(), e.to_string()),
            }
        } else if tool_name == "check_syntax" {
            let file_path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            match crate::tools::validation_tools::check_syntax(file_path).await {
                Ok(mut result) => {
                    result.tool_use_id = tool_use_id.to_string();
                    result
                }
                Err(e) => ToolResult::error(tool_use_id.to_string(), e.to_string()),
            }
        } else {
            ToolResult::error(
                tool_use_id.to_string(),
                format!("Unknown tool category for: {}", tool_name),
            )
        }
    }

    /// Change permission mode
    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.permission_mode = mode;
    }

    /// Get current permission mode
    pub fn permission_mode(&self) -> PermissionMode {
        self.permission_mode
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self {
            registry: brainwires_tool_builtins::registry_with_builtins(),
            permission_mode: PermissionMode::Auto,
            provider: None,
            orchestrator: OrchestratorTool::new(),
            orchestrator_initialized: false,
            access_control: None,
            agent_pool: None,
            agent_pool_tool: None,
            task_manager: None,
            task_manager_tool: None,
            session_task_tool: None,
            monitor_tool: MonitorTool::new(),
            policy_engine: None,
            audit_logger: None,
            approval_tx: None,
            session_approvals: std::collections::HashMap::new(),
            sudo_password_tx: None,
            remote_bridge: None,
            org_blocked_tools: Vec::new(),
            org_permission_relay_required: false,
            org_audit_all_commands: false,
            settings: None,
            hooks: None,
            user_question_tx: None,
        }
    }
}

// ── Framework trait implementation ──────────────────────────────────────────

/// Implement the framework's `ToolExecutor` trait so that the CLI's concrete
/// executor can be used with `brainwires-agent`' `TaskAgent` and `AgentPool`.
#[async_trait::async_trait]
impl brainwires::tools::ToolExecutor for ToolExecutor {
    async fn execute(
        &self,
        tool_use: &crate::types::tool::ToolUse,
        context: &crate::types::tool::ToolContext,
    ) -> anyhow::Result<crate::types::tool::ToolResult> {
        // Delegate to the existing concrete implementation.
        ToolExecutor::execute(self, tool_use, context).await
    }

    fn available_tools(&self) -> Vec<crate::types::tool::Tool> {
        self.registry.get_all().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_registry() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let tools = executor.get_tools();

        // Should have tools registered
        assert!(!tools.is_empty());

        // Check for specific tools
        assert!(tools.iter().any(|t| t.name == "read_file"));
        assert!(tools.iter().any(|t| t.name == "execute_command"));
        assert!(tools.iter().any(|t| t.name == "git_status"));
    }

    #[tokio::test]
    async fn test_execute_read_only_mode() {
        let executor = ToolExecutor::new(PermissionMode::ReadOnly);
        let context = ToolContext::default();

        let tool_use = ToolUse {
            id: "test-1".to_string(),
            name: "write_file".to_string(),
            input: serde_json::json!({
                "path": "/tmp/test.txt",
                "content": "test"
            }),
        };

        let result = executor.execute(&tool_use, &context).await.unwrap();
        assert!(result.is_error);
    }

    /// Smoke test: a `deny` rule in layered settings blocks tool execution
    /// even under `PermissionMode::Full`. This is the central safety
    /// guarantee documented in `docs/harness/settings.md`.
    #[tokio::test]
    async fn settings_deny_blocks_even_in_full_mode() {
        let mut executor = ToolExecutor::new(PermissionMode::Full);
        let settings = std::sync::Arc::new(crate::config::Settings {
            permissions: Some(crate::config::Permissions {
                allow: vec![],
                deny: vec!["Bash(rm:*)".into()],
                ask: vec![],
            }),
            ..Default::default()
        });
        executor.set_settings(settings);

        let tool_use = ToolUse {
            id: "deny-test".to_string(),
            name: "execute_command".to_string(),
            input: serde_json::json!({"command": "rm -rf /tmp/bwsmoke"}),
        };
        let context = ToolContext::default();
        let result = executor.execute(&tool_use, &context).await.unwrap();
        assert!(result.is_error, "expected deny to produce an error result");
        assert!(
            result.content.contains("denied by settings rule"),
            "error should cite the deny rule, got: {}",
            result.content
        );
    }

    #[test]
    fn test_tool_executor_default() {
        let executor = ToolExecutor::default();
        assert_eq!(executor.permission_mode(), PermissionMode::Auto);
    }

    #[test]
    fn test_set_permission_mode() {
        let mut executor = ToolExecutor::new(PermissionMode::Auto);
        assert_eq!(executor.permission_mode(), PermissionMode::Auto);

        executor.set_permission_mode(PermissionMode::Full);
        assert_eq!(executor.permission_mode(), PermissionMode::Full);

        executor.set_permission_mode(PermissionMode::ReadOnly);
        assert_eq!(executor.permission_mode(), PermissionMode::ReadOnly);
    }

    #[test]
    fn test_get_tools_not_empty() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let tools = executor.get_tools();
        assert!(tools.len() > 5, "Should have multiple tools registered");
    }

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let tool_use = ToolUse {
            id: "test-2".to_string(),
            name: "nonexistent_tool".to_string(),
            input: serde_json::json!({}),
        };

        let result = executor.execute(&tool_use, &context).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_auto_mode() {
        let executor = ToolExecutor::new(PermissionMode::Auto);
        let context = ToolContext::default();

        let tool_use = ToolUse {
            id: "test-3".to_string(),
            name: "write_file".to_string(),
            input: serde_json::json!({
                "path": "/tmp/test_auto.txt",
                "content": "auto mode test"
            }),
        };

        // In auto mode, tools requiring approval should succeed
        let result = executor.execute(&tool_use, &context).await.unwrap();
        // Result may error due to actual execution, but shouldn't be blocked by permission
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn test_route_tool_file_ops() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        // Test all file operation tools
        let file_tools = vec![
            "read_file",
            "write_file",
            "edit_file",
            "list_directory",
            "search_files",
            "delete_file",
            "create_directory",
        ];

        for tool in file_tools {
            let result = executor
                .route_tool("test-tool-id", tool, &serde_json::json!({}), &context)
                .await;
            // Should return a result (may be error due to missing params, but shouldn't be unknown category)
            assert!(!result.content.contains("Unknown tool category"));
        }
    }

    #[tokio::test]
    async fn test_route_tool_bash() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let result = executor
            .route_tool(
                "test-bash-id",
                "execute_command",
                &serde_json::json!({}),
                &context,
            )
            .await;
        assert!(!result.content.contains("Unknown tool category"));
    }

    #[tokio::test]
    async fn test_route_tool_git() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let git_tools = vec!["git_status", "git_log", "git_diff", "git_commit"];

        for tool in git_tools {
            let result = executor
                .route_tool("test-git-id", tool, &serde_json::json!({}), &context)
                .await;
            assert!(!result.content.contains("Unknown tool category"));
        }
    }

    #[tokio::test]
    async fn test_route_tool_web() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let result = executor
            .route_tool("test-web-id", "fetch_url", &serde_json::json!({}), &context)
            .await;
        assert!(!result.content.contains("Unknown tool category"));
    }

    #[tokio::test]
    async fn test_route_tool_search() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let result = executor
            .route_tool(
                "test-search-id",
                "search_code",
                &serde_json::json!({}),
                &context,
            )
            .await;
        assert!(!result.content.contains("Unknown tool category"));
    }

    // Note: semantic_search tools test removed due to nested runtime issues
    // The routing logic is tested through the unknown_category test

    #[tokio::test]
    #[ignore = "MCP tool causes nested runtime issue in test environment"]
    async fn test_route_tool_mcp() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let result = executor
            .route_tool(
                "test-mcp-id",
                "mcp_test_tool",
                &serde_json::json!({}),
                &context,
            )
            .await;
        assert!(!result.content.contains("Unknown tool category"));
    }

    #[tokio::test]
    async fn test_route_tool_unknown_category() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let result = executor
            .route_tool(
                "test-unknown-id",
                "completely_unknown",
                &serde_json::json!({}),
                &context,
            )
            .await;
        assert!(result.is_error);
        assert!(result.content.contains("Unknown tool category"));
    }

    #[tokio::test]
    async fn test_execute_full_mode() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let context = ToolContext::default();

        let tool_use = ToolUse {
            id: "test-full".to_string(),
            name: "write_file".to_string(),
            input: serde_json::json!({
                "path": "/tmp/test_full.txt",
                "content": "full mode test"
            }),
        };

        // In full mode, all tools should be allowed
        let result = executor.execute(&tool_use, &context).await.unwrap();
        assert!(!result.content.is_empty());
    }

    #[test]
    fn test_permission_mode_getter() {
        let executor_auto = ToolExecutor::new(PermissionMode::Auto);
        assert_eq!(executor_auto.permission_mode(), PermissionMode::Auto);

        let executor_full = ToolExecutor::new(PermissionMode::Full);
        assert_eq!(executor_full.permission_mode(), PermissionMode::Full);

        let executor_readonly = ToolExecutor::new(PermissionMode::ReadOnly);
        assert_eq!(
            executor_readonly.permission_mode(),
            PermissionMode::ReadOnly
        );
    }

    #[test]
    fn test_new_with_different_modes() {
        let auto = ToolExecutor::new(PermissionMode::Auto);
        assert_eq!(auto.permission_mode(), PermissionMode::Auto);

        let full = ToolExecutor::new(PermissionMode::Full);
        assert_eq!(full.permission_mode(), PermissionMode::Full);

        let readonly = ToolExecutor::new(PermissionMode::ReadOnly);
        assert_eq!(readonly.permission_mode(), PermissionMode::ReadOnly);
    }

    #[test]
    fn test_get_tools_has_expected_tools() {
        let executor = ToolExecutor::new(PermissionMode::Full);
        let tools = executor.get_tools();

        // Check for expected tool names
        let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();

        assert!(tool_names.contains(&"read_file".to_string()));
        assert!(tool_names.contains(&"write_file".to_string()));
        assert!(tool_names.contains(&"execute_command".to_string()));
        assert!(tool_names.contains(&"git_status".to_string()));
        assert!(tool_names.contains(&"fetch_url".to_string()));
    }

    // === Approval System Tests ===

    #[test]
    fn test_tool_requires_approval() {
        let executor = ToolExecutor::new(PermissionMode::Auto);

        // Tools that should require approval
        assert!(executor.tool_requires_approval("write_file"));
        assert!(executor.tool_requires_approval("edit_file"));
        assert!(executor.tool_requires_approval("patch_file"));
        assert!(executor.tool_requires_approval("delete_file"));
        assert!(executor.tool_requires_approval("create_directory"));
        assert!(executor.tool_requires_approval("execute_command"));
        assert!(executor.tool_requires_approval("git_commit"));
        assert!(executor.tool_requires_approval("git_push"));
        assert!(executor.tool_requires_approval("git_reset"));
        assert!(executor.tool_requires_approval("git_checkout"));

        // Tools that should NOT require approval
        assert!(!executor.tool_requires_approval("read_file"));
        assert!(!executor.tool_requires_approval("list_directory"));
        assert!(!executor.tool_requires_approval("search_files"));
        assert!(!executor.tool_requires_approval("git_status"));
        assert!(!executor.tool_requires_approval("git_log"));
        assert!(!executor.tool_requires_approval("git_diff"));
        assert!(!executor.tool_requires_approval("fetch_url"));
    }

    #[test]
    fn test_create_approval_action_write_file() {
        use crate::approval::ApprovalAction;

        let executor = ToolExecutor::new(PermissionMode::Auto);
        let input = serde_json::json!({
            "path": "/tmp/test.txt",
            "content": "hello"
        });

        let action = executor.create_approval_action("write_file", &input);
        match action {
            ApprovalAction::WriteFile { path } => {
                assert_eq!(path, "/tmp/test.txt");
            }
            _ => panic!("Expected WriteFile action"),
        }
    }

    #[test]
    fn test_create_approval_action_edit_file() {
        use crate::approval::ApprovalAction;

        let executor = ToolExecutor::new(PermissionMode::Auto);
        let input = serde_json::json!({
            "path": "/tmp/edit.rs",
            "old_string": "foo",
            "new_string": "bar"
        });

        let action = executor.create_approval_action("edit_file", &input);
        match action {
            ApprovalAction::EditFile { path } => {
                assert_eq!(path, "/tmp/edit.rs");
            }
            _ => panic!("Expected EditFile action"),
        }
    }

    #[test]
    fn test_create_approval_action_delete_file() {
        use crate::approval::ApprovalAction;

        let executor = ToolExecutor::new(PermissionMode::Auto);
        let input = serde_json::json!({
            "path": "/tmp/delete_me.txt"
        });

        let action = executor.create_approval_action("delete_file", &input);
        match action {
            ApprovalAction::DeleteFile { path } => {
                assert_eq!(path, "/tmp/delete_me.txt");
            }
            _ => panic!("Expected DeleteFile action"),
        }
    }

    #[test]
    fn test_create_approval_action_execute_command() {
        use crate::approval::ApprovalAction;

        let executor = ToolExecutor::new(PermissionMode::Auto);
        let input = serde_json::json!({
            "command": "ls -la"
        });

        let action = executor.create_approval_action("execute_command", &input);
        match action {
            ApprovalAction::ExecuteCommand { command } => {
                assert_eq!(command, "ls -la");
            }
            _ => panic!("Expected ExecuteCommand action"),
        }
    }

    #[test]
    fn test_create_approval_action_git_modify() {
        use crate::approval::ApprovalAction;

        let executor = ToolExecutor::new(PermissionMode::Auto);
        let input = serde_json::json!({
            "message": "test commit"
        });

        let action = executor.create_approval_action("git_commit", &input);
        match action {
            ApprovalAction::GitModify { operation } => {
                assert_eq!(operation, "commit");
            }
            _ => panic!("Expected GitModify action"),
        }
    }

    #[test]
    fn test_approval_channel_setup() {
        let mut executor = ToolExecutor::new(PermissionMode::Auto);
        assert!(!executor.has_approval_channel());

        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::approval::ApprovalRequest>(16);
        executor.set_approval_channel(tx);
        assert!(executor.has_approval_channel());
    }

    #[test]
    fn test_session_approval_recording() {
        use crate::approval::ApprovalResponse;

        let mut executor = ToolExecutor::new(PermissionMode::Auto);

        // No session decision initially
        assert!(executor.get_session_approval("write_file").is_none());

        // Record session approval
        executor.record_session_approval("write_file", ApprovalResponse::ApproveForSession);
        assert_eq!(
            executor.get_session_approval("write_file"),
            Some(ApprovalResponse::ApproveForSession)
        );

        // Non-session responses shouldn't be recorded
        executor.record_session_approval("delete_file", ApprovalResponse::Approve);
        assert!(executor.get_session_approval("delete_file").is_none());

        // Session deny should be recorded
        executor.record_session_approval("execute_command", ApprovalResponse::DenyForSession);
        assert_eq!(
            executor.get_session_approval("execute_command"),
            Some(ApprovalResponse::DenyForSession)
        );
    }

    #[tokio::test]
    async fn test_approval_flow_no_channel() {
        use crate::approval::ApprovalResponse;

        let executor = ToolExecutor::new(PermissionMode::Auto);

        // Without a channel, approval should default to Approve
        let response = executor
            .request_approval(
                "write_file",
                "Write a file",
                &serde_json::json!({"path": "/tmp/test.txt"}),
            )
            .await
            .unwrap();

        assert_eq!(response, ApprovalResponse::Approve);
    }

    #[tokio::test]
    async fn test_approval_flow_with_session_decision() {
        use crate::approval::ApprovalResponse;

        let mut executor = ToolExecutor::new(PermissionMode::Auto);

        // Pre-record a session decision
        executor.session_approvals.insert(
            "write_file".to_string(),
            ApprovalResponse::ApproveForSession,
        );

        // Request should return the session decision without prompting
        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::approval::ApprovalRequest>(16);
        executor.set_approval_channel(tx);

        let response = executor
            .request_approval(
                "write_file",
                "Write a file",
                &serde_json::json!({"path": "/tmp/test.txt"}),
            )
            .await
            .unwrap();

        assert_eq!(response, ApprovalResponse::ApproveForSession);
    }

    #[tokio::test]
    async fn test_approval_flow_interactive() {
        use crate::approval::{ApprovalRequest, ApprovalResponse};

        let mut executor = ToolExecutor::new(PermissionMode::Auto);
        let (tx, mut rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(16);
        executor.set_approval_channel(tx);

        // Spawn a task to handle the approval request
        let handle = tokio::spawn(async move {
            // Wait for the request
            let request = rx.recv().await.expect("Should receive request");
            assert_eq!(request.tool_name, "write_file");

            // Send approval response
            request.response_tx.send(ApprovalResponse::Approve).unwrap();
        });

        // Request approval
        let response = executor
            .request_approval(
                "write_file",
                "Write a file",
                &serde_json::json!({"path": "/tmp/test.txt"}),
            )
            .await
            .unwrap();

        assert_eq!(response, ApprovalResponse::Approve);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_approval_flow_deny() {
        use crate::approval::{ApprovalRequest, ApprovalResponse};

        let mut executor = ToolExecutor::new(PermissionMode::Auto);
        let (tx, mut rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(16);
        executor.set_approval_channel(tx);

        // Spawn a task to deny the request
        let handle = tokio::spawn(async move {
            let request = rx.recv().await.expect("Should receive request");
            request.response_tx.send(ApprovalResponse::Deny).unwrap();
        });

        let response = executor
            .request_approval(
                "delete_file",
                "Delete a file",
                &serde_json::json!({"path": "/tmp/important.txt"}),
            )
            .await
            .unwrap();

        assert_eq!(response, ApprovalResponse::Deny);
        handle.await.unwrap();
    }
}
