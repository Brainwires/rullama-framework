//! Command Handler
//!
//! Handles slash command parsing and execution.

use super::super::state::{App, AppMode, ApprovalMode, LogLevel, TuiMessage};
use crate::providers::ProviderFactory;
use crate::types::message::{Message, MessageContent, Role};
use anyhow::Result;

impl App {
    /// Handle slash command execution
    /// Returns true if AI processing should be skipped (command was a pure action/help/error)
    /// Returns false if AI should be called (command produced a message to send)
    pub(super) async fn handle_command(
        &mut self,
        cmd_name: String,
        cmd_args: &[String],
        _user_content: String,
    ) -> Result<bool> {
        use crate::commands::executor::CommandResult;

        // Execute slash command
        match self.command_executor.execute(&cmd_name, cmd_args) {
            Ok(CommandResult::Help(lines)) => {
                // Display help as system message
                let help_text = lines.join("\n");
                self.messages.push(TuiMessage {
                    role: "system".to_string(),
                    content: help_text,
                    created_at: chrono::Utc::now().timestamp(),
                });
                self.clear_input();
                Ok(true) // Skip AI
            }
            Ok(CommandResult::Action(action)) => {
                self.handle_command_action(action).await?;
                Ok(true) // Skip AI - actions don't need AI response
            }
            Ok(CommandResult::ActionWithMessage(action, msg)) => {
                // Execute the action (e.g., switch prompt mode), then send the message to AI
                self.handle_command_action(action).await?;

                // Add the message as user input for AI processing
                let user_message = TuiMessage {
                    role: "user".to_string(),
                    content: msg.clone(),
                    created_at: chrono::Utc::now().timestamp(),
                };
                self.messages.push(user_message);

                self.conversation_history.push(Message {
                    role: Role::User,
                    content: MessageContent::Text(msg),
                    name: None,
                    metadata: None,
                });
                Ok(false) // Don't skip AI - need to process the message
            }
            Ok(CommandResult::Message(msg)) => {
                // Command produced a message to send to AI
                // Use the expanded message instead of original input
                let user_message = TuiMessage {
                    role: "user".to_string(),
                    content: msg.clone(),
                    created_at: chrono::Utc::now().timestamp(),
                };
                self.messages.push(user_message);

                self.conversation_history.push(Message {
                    role: Role::User,
                    content: MessageContent::Text(msg.clone()),
                    name: None,
                    metadata: None,
                });
                // Continue to AI processing
                Ok(false) // Don't skip AI - need to process the message
            }
            Err(e) => {
                // Fall-through: if the unknown command matches a discovered
                // skill, treat `/<skill-name> args` as `/skill <name> args`.
                // Mirrors Claude Code's skill-invocation shorthand.
                let matches_skill = self
                    .skill_registry
                    .as_ref()
                    .map(|r| r.contains(&cmd_name))
                    .unwrap_or(false);
                if matches_skill {
                    self.handle_invoke_skill(&cmd_name, cmd_args.to_vec()).await;
                    return Ok(true);
                }

                // Display error as system message
                self.messages.push(TuiMessage {
                    role: "system".to_string(),
                    content: format!("Command error: {}", e),
                    created_at: chrono::Utc::now().timestamp(),
                });
                self.clear_input();
                Ok(true) // Skip AI
            }
        }
    }

    /// Handle command action, returns true if processing should stop
    pub(super) async fn handle_command_action(
        &mut self,
        action: crate::commands::executor::CommandAction,
    ) -> Result<bool> {
        use crate::commands::executor::CommandAction;

        match action {
            CommandAction::ClearHistory => {
                // Save current state before clearing (for /resume)
                self.cleared_messages = Some(self.messages.clone());
                self.cleared_conversation_history = Some(self.conversation_history.clone());

                // Clear conversation history
                self.messages.clear();
                self.conversation_history.clear();
                // Also clear shell history
                self.shell_history.clear();
                self.selected_shell_index = 0;
                self.shell_viewer_scroll = 0;

                self.set_status(
                    LogLevel::Info,
                    "Conversation and shell history cleared (use /resume to restore)",
                );
                self.clear_input();
                Ok(true)
            }
            CommandAction::ResumeHistory(conversation_id) => {
                use super::super::session_management::SessionManagement;
                if let Some(conv_id) = conversation_id {
                    // Load conversation from database by ID
                    match self.load_conversation(&conv_id).await {
                        Ok(()) => {
                            self.set_status(
                                LogLevel::Info,
                                format!("Loaded conversation: {}", conv_id),
                            );
                        }
                        Err(e) => {
                            self.set_status(
                                LogLevel::Error,
                                format!("Failed to load conversation: {}", e),
                            );
                        }
                    }
                } else {
                    // Show session picker to select a conversation
                    match self.conversation_store.list(Some(50)).await {
                        Ok(conversations) => {
                            if conversations.is_empty() {
                                self.set_status(LogLevel::Info, "No saved conversations found");
                            } else {
                                // list() already returns sorted by updated_at descending (newest first)
                                self.available_sessions = conversations;
                                self.selected_session_index = 0;
                                self.session_picker_scroll = 0;
                                self.mode = AppMode::SessionPicker;
                                self.set_status(LogLevel::Info, "Select a conversation to resume (↑/↓ to navigate, Enter to select, Esc to cancel)");
                            }
                        }
                        Err(e) => {
                            self.set_status(
                                LogLevel::Error,
                                format!("Failed to load conversations: {}", e),
                            );
                        }
                    }
                }
                self.clear_input();
                Ok(true)
            }
            CommandAction::SwitchModel(new_model) => {
                // Switch model by recreating the provider
                match ProviderFactory::new().create(new_model.clone()).await {
                    Ok(new_provider) => {
                        self.provider = new_provider;
                        self.model = new_model.clone();

                        // Persist the model selection to config
                        if let Err(e) = Self::update_config_model(&new_model) {
                            tracing::warn!("Failed to persist model to config: {}", e);
                        }

                        self.set_status(
                            LogLevel::Info,
                            format!("Model switched to: {}", new_model),
                        );
                    }
                    Err(e) => {
                        self.set_status(LogLevel::Error, format!("Failed to switch model: {}", e));
                    }
                }
                self.clear_input();
                Ok(true)
            }
            CommandAction::SwitchProvider(name) => {
                self.handle_switch_provider(name).await;
                self.clear_input();
                Ok(true)
            }
            CommandAction::ListProviders => {
                self.handle_list_providers();
                self.clear_input();
                Ok(true)
            }
            CommandAction::ShowStatus => {
                // Show status as system message
                let status_msg = format!(
                    "Session: {}\nModel: {}\nMessages: {}",
                    self.session_id,
                    self.model,
                    self.messages.len()
                );
                self.messages.push(TuiMessage {
                    role: "system".to_string(),
                    content: status_msg,
                    created_at: chrono::Utc::now().timestamp(),
                });
                self.clear_input();
                Ok(true)
            }
            CommandAction::Rewind(steps) => {
                // Rewind conversation
                let remove_count = (steps * 2).min(self.messages.len());
                for _ in 0..remove_count {
                    self.messages.pop();
                    self.conversation_history.pop();
                }
                self.set_status(LogLevel::Info, format!("Rewound {} steps", steps));
                self.clear_input();
                Ok(true)
            }
            CommandAction::CreateCheckpoint(name) => {
                self.handle_create_checkpoint(name).await?;
                Ok(true)
            }
            CommandAction::RestoreCheckpoint(checkpoint_id) => {
                self.handle_restore_checkpoint(checkpoint_id).await?;
                Ok(true)
            }
            CommandAction::ListCheckpoints => {
                self.handle_list_checkpoints().await?;
                Ok(true)
            }
            CommandAction::Exit => {
                // Exit the application
                self.should_quit = true;
                self.clear_input();
                Ok(true)
            }
            CommandAction::SetApprovalMode(mode) => {
                // Set approval mode
                self.approval_mode = match mode.as_str() {
                    "suggest" => ApprovalMode::Suggest,
                    "auto-edit" => ApprovalMode::AutoEdit,
                    "full-auto" => ApprovalMode::FullAuto,
                    _ => ApprovalMode::Suggest, // Default to safest
                };
                self.set_status(LogLevel::Info, format!("Approval mode set to: {}", mode));
                self.clear_input();
                Ok(true)
            }
            CommandAction::ExecCommand(command) => {
                // Store command to be executed in main loop where we have terminal access
                self.pending_exec_command = Some(command);
                self.clear_input();
                Ok(true)
            }
            CommandAction::OpenShell => {
                // Main loop takes over and hands the terminal to an
                // interactive shell; we just raise the flag here.
                #[cfg(unix)]
                {
                    self.pending_shell = true;
                    self.add_console_message(
                        "Opening interactive shell (exit/Ctrl+D to return)...".to_string(),
                    );
                }
                #[cfg(not(unix))]
                {
                    self.add_console_message(
                        "/shell is not supported on this platform yet".to_string(),
                    );
                }
                self.clear_input();
                Ok(true)
            }
            CommandAction::ShowShellHistory => {
                // Open shell history viewer
                self.mode = AppMode::ShellViewer;
                self.selected_shell_index = self.shell_history.len().saturating_sub(1);
                self.shell_viewer_scroll = 0;
                self.clear_input();
                Ok(true)
            }
            CommandAction::OpenHotkeyDialog => {
                // Open hotkey configuration dialog
                use ratatui_interact::components::hotkey_dialog::HotkeyDialogState;
                self.hotkey_dialog_state = Some(HotkeyDialogState::new());
                self.mode = AppMode::HotkeyDialog;
                self.clear_input();
                Ok(true)
            }
            CommandAction::ListPlans(conversation_id) => {
                self.handle_list_plans(conversation_id).await?;
                Ok(true)
            }
            CommandAction::ShowPlan(plan_id) => {
                self.handle_show_plan(plan_id).await?;
                Ok(true)
            }
            CommandAction::DeletePlan(plan_id) => {
                self.handle_delete_plan(plan_id).await?;
                Ok(true)
            }
            CommandAction::ActivatePlan(plan_id) => {
                self.handle_activate_plan(plan_id).await?;
                Ok(true)
            }
            CommandAction::DeactivatePlan => {
                self.handle_deactivate_plan();
                Ok(true)
            }
            CommandAction::PlanStatus => {
                self.handle_plan_status();
                Ok(true)
            }
            CommandAction::PausePlan => {
                self.handle_pause_plan().await?;
                Ok(true)
            }
            CommandAction::ResumePlan(plan_id) => {
                self.handle_resume_plan(plan_id).await?;
                Ok(true)
            }
            CommandAction::ShowTasks => {
                self.handle_show_tasks().await;
                Ok(true)
            }
            CommandAction::TaskComplete(task_id) => {
                self.handle_task_complete(task_id).await;
                Ok(true)
            }
            CommandAction::TaskSkip(task_id, reason) => {
                self.handle_task_skip(task_id, reason).await;
                Ok(true)
            }
            CommandAction::TaskAdd(description) => {
                self.handle_task_add(description).await;
                Ok(true)
            }
            CommandAction::TaskStart(task_id) => {
                self.handle_task_start(task_id).await;
                Ok(true)
            }
            CommandAction::TaskBlock(task_id, reason) => {
                self.handle_task_block(task_id, reason).await;
                Ok(true)
            }
            CommandAction::TaskDepends(task_id, depends_on) => {
                self.handle_task_depends(task_id, depends_on).await;
                Ok(true)
            }
            CommandAction::TaskReady => {
                self.handle_task_ready().await;
                Ok(true)
            }
            CommandAction::TaskTime(task_id) => {
                self.handle_task_time(task_id).await;
                Ok(true)
            }
            CommandAction::TaskList => {
                self.handle_task_list().await;
                Ok(true)
            }
            CommandAction::ExecutePlan(plan_id, mode) => {
                self.handle_execute_plan(plan_id, mode).await?;
                Ok(true)
            }
            CommandAction::ListTemplates => {
                self.handle_list_templates().await;
                Ok(true)
            }
            CommandAction::SaveTemplate(name, description) => {
                self.handle_save_template(name, description).await?;
                Ok(true)
            }
            CommandAction::ShowTemplate(name) => {
                self.handle_show_template(name).await;
                Ok(true)
            }
            CommandAction::UseTemplate(name, vars) => {
                self.handle_use_template(name, vars).await?;
                Ok(true)
            }
            CommandAction::DeleteTemplate(name) => {
                self.handle_delete_template(name).await;
                Ok(true)
            }
            CommandAction::SearchPlans(query) => {
                self.handle_search_plans(query).await;
                Ok(true)
            }
            CommandAction::BranchPlan(name, task) => {
                self.handle_branch_plan(name, task).await?;
                Ok(true)
            }
            CommandAction::MergePlan(plan_id) => {
                self.handle_merge_plan(plan_id).await?;
                Ok(true)
            }
            CommandAction::PlanTree(plan_id) => {
                self.handle_plan_tree(plan_id).await;
                Ok(true)
            }
            // Context/Working Set commands
            CommandAction::ContextShow => {
                self.add_console_message(self.working_set.display());
                Ok(true)
            }
            CommandAction::ContextAdd(path, pinned) => {
                self.handle_context_add(&path, pinned);
                Ok(true)
            }
            CommandAction::ContextRemove(path) => {
                self.handle_context_remove(&path);
                Ok(true)
            }
            CommandAction::ContextPin(path) => {
                self.handle_context_pin(&path);
                Ok(true)
            }
            CommandAction::ContextUnpin(path) => {
                self.handle_context_unpin(&path);
                Ok(true)
            }
            CommandAction::ContextClear(keep_pinned) => {
                self.handle_context_clear(keep_pinned);
                Ok(true)
            }
            // Tool mode commands
            CommandAction::ShowToolMode => {
                self.handle_show_tool_mode();
                Ok(true)
            }
            CommandAction::SetToolMode(mode) => {
                self.handle_set_tool_mode(mode);
                Ok(true)
            }
            CommandAction::OpenToolPicker => {
                self.handle_open_tool_picker();
                Ok(true)
            }
            // MDAP commands
            CommandAction::MdapStatus => {
                self.handle_mdap_status();
                Ok(true)
            }
            CommandAction::MdapEnable => {
                self.handle_mdap_enable();
                Ok(true)
            }
            CommandAction::MdapDisable => {
                self.handle_mdap_disable();
                Ok(true)
            }
            CommandAction::MdapSetK(k) => {
                self.handle_mdap_set_k(k);
                Ok(true)
            }
            CommandAction::MdapSetTarget(target) => {
                self.handle_mdap_set_target(target);
                Ok(true)
            }
            // Dream (sleep) consolidation commands
            CommandAction::DreamRun => {
                self.handle_dream_run().await;
                Ok(true)
            }
            CommandAction::DreamStatus => {
                self.handle_dream_status();
                Ok(true)
            }
            // Knowledge commands
            CommandAction::LearnTruth(rule, rationale) => {
                self.handle_learn_truth(&rule, rationale.as_deref()).await;
                Ok(true)
            }
            CommandAction::KnowledgeStatus => {
                self.handle_knowledge_status().await;
                Ok(true)
            }
            CommandAction::KnowledgeList(category) => {
                self.handle_knowledge_list(category.as_deref()).await;
                Ok(true)
            }
            CommandAction::KnowledgeSearch(query) => {
                self.handle_knowledge_search(&query).await;
                Ok(true)
            }
            CommandAction::KnowledgeSync => {
                self.handle_knowledge_sync().await;
                Ok(true)
            }
            CommandAction::KnowledgeContradict(id, reason) => {
                self.handle_knowledge_contradict(&id, reason.as_deref())
                    .await;
                Ok(true)
            }
            CommandAction::KnowledgeDelete(id) => {
                self.handle_knowledge_delete(&id).await;
                Ok(true)
            }
            // Personal Knowledge System commands
            CommandAction::ProfileShow => {
                self.handle_profile_show().await;
                Ok(true)
            }
            CommandAction::ProfileSet(key, value, local_only) => {
                self.handle_profile_set(&key, &value, local_only).await;
                Ok(true)
            }
            CommandAction::ProfileList(category) => {
                self.handle_profile_list(category.as_deref()).await;
                Ok(true)
            }
            CommandAction::ProfileSearch(query) => {
                self.handle_profile_search(&query).await;
                Ok(true)
            }
            CommandAction::ProfileDelete(id_or_key) => {
                self.handle_profile_delete(&id_or_key).await;
                Ok(true)
            }
            CommandAction::ProfileSync => {
                self.handle_profile_sync().await;
                Ok(true)
            }
            CommandAction::ProfileExport(path) => {
                self.handle_profile_export(path.as_deref()).await;
                Ok(true)
            }
            CommandAction::ProfileImport(path) => {
                self.handle_profile_import(&path).await;
                Ok(true)
            }
            CommandAction::ProfileStats => {
                self.handle_profile_stats().await;
                Ok(true)
            }
            // Multi-Agent System Actions
            CommandAction::ListAgents => {
                self.handle_list_agents().await;
                Ok(true)
            }
            CommandAction::SwitchAgent(session_id) => {
                self.handle_switch_agent(&session_id).await;
                Ok(true)
            }
            CommandAction::SpawnChildAgent(model, reason) => {
                self.handle_spawn_child_agent(model, reason).await;
                Ok(true)
            }
            CommandAction::AgentTree => {
                self.handle_agent_tree().await;
                Ok(true)
            }
            CommandAction::HibernateAgents => {
                self.handle_hibernate_agents().await;
                Ok(true)
            }
            CommandAction::ResumeAgents => {
                self.handle_resume_agents().await;
                Ok(true)
            }
            // Skill commands
            CommandAction::InvokeSkill(name, args) => {
                self.handle_invoke_skill(&name, args).await;
                Ok(true)
            }
            CommandAction::ListSkills => {
                self.handle_list_skills().await;
                Ok(true)
            }
            CommandAction::ShowSkill(name) => {
                self.handle_show_skill(&name).await;
                Ok(true)
            }
            CommandAction::ReloadSkills => {
                self.handle_reload_skills().await;
                Ok(true)
            }
            CommandAction::CreateSkill(name, location) => {
                self.handle_create_skill(&name, location.as_deref()).await;
                Ok(true)
            }
            // Prompt mode commands
            CommandAction::SetPromptModeAsk => {
                self.set_prompt_mode_ask().await?;
                self.clear_input();
                Ok(true)
            }
            CommandAction::SetPromptModeEdit => {
                self.set_prompt_mode_edit().await?;
                self.clear_input();
                Ok(true)
            }
            // Plan mode commands
            CommandAction::EnterPlanMode(focus) => {
                self.enter_plan_mode(focus).await?;
                Ok(true)
            }
            CommandAction::ExitPlanMode => {
                self.exit_plan_mode().await?;
                Ok(true)
            }
            CommandAction::PlanModeStatus => {
                let status = self.plan_mode_status();
                self.add_console_message(status);
                self.clear_input();
                Ok(true)
            }
            CommandAction::ClearPlanMode => {
                self.clear_plan_mode();
                self.clear_input();
                Ok(true)
            }
            CommandAction::ExportPlanMode(path) => {
                // Export plan mode session to file
                if let Some(ref state) = self.plan_mode_state {
                    let output = if let Some(p) = path {
                        p
                    } else {
                        format!("plan-{}.md", state.plan_session_id)
                    };
                    let content = state
                        .messages
                        .iter()
                        .map(|m| format!("## {}\n\n{}\n", m.role, m.content))
                        .collect::<Vec<_>>()
                        .join("\n---\n\n");
                    match std::fs::write(&output, &content) {
                        Ok(_) => {
                            self.add_console_message(format!("Exported plan mode to: {}", output))
                        }
                        Err(e) => self.add_console_message(format!("Failed to export: {}", e)),
                    }
                } else {
                    self.add_console_message("No plan mode session to export".to_string());
                }
                self.clear_input();
                Ok(true)
            }
        }
    }

    /// Handle /mdap (show status)
    fn handle_mdap_status(&mut self) {
        let status_msg = if let Some(ref config) = self.mdap_config {
            format!(
                "MDAP Mode: Enabled\n\n\
                Configuration:\n\
                - Vote margin (k): {}\n\
                - Target success rate: {:.1}%\n\
                - Parallel samples: {}\n\
                - Max samples/subtask: {}\n\n\
                Commands:\n\
                - /mdap off      - Disable MDAP mode\n\
                - /mdap:k <n>    - Set vote margin\n\
                - /mdap:target <rate> - Set target success rate",
                config.k,
                config.target_success_rate * 100.0,
                config.parallel_samples,
                config.max_samples_per_subtask
            )
        } else {
            "MDAP Mode: Disabled\n\n\
            MDAP (Massively Decomposed Agentic Processes) enables high-reliability\n\
            execution through task decomposition and multi-sample voting.\n\n\
            Commands:\n\
            - /mdap on       - Enable MDAP mode\n\
            - /mdap:k <n>    - Set vote margin (1-10, default: 3)\n\
            - /mdap:target <rate> - Set target success rate (default: 0.95)"
                .to_string()
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: status_msg,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /mdap:on
    fn handle_mdap_enable(&mut self) {
        use crate::mdap::MdapConfig;

        if self.mdap_config.is_some() {
            self.add_console_message("ℹ️  MDAP mode is already enabled".to_string());
        } else {
            self.mdap_config = Some(MdapConfig::default());
            self.set_status(
                LogLevel::Info,
                format!("Ready - Model: {} [MDAP] (Ctrl+C to quit)", self.model),
            );
            self.add_console_message("✅ MDAP mode enabled (k=3, target=95%)".to_string());
        }
        self.clear_input();
    }

    /// Handle /mdap:off
    fn handle_mdap_disable(&mut self) {
        if self.mdap_config.is_some() {
            self.mdap_config = None;
            self.set_status(
                LogLevel::Info,
                format!("Ready - Model: {} (Ctrl+C to quit)", self.model),
            );
            self.add_console_message("✅ MDAP mode disabled".to_string());
        } else {
            self.add_console_message("ℹ️  MDAP mode is already disabled".to_string());
        }
        self.clear_input();
    }

    /// Handle /mdap:k
    fn handle_mdap_set_k(&mut self, k: u32) {
        use crate::mdap::MdapConfig;

        if let Some(ref mut config) = self.mdap_config {
            config.k = k;
            self.add_console_message(format!("✅ MDAP vote margin set to k={}", k));
        } else {
            // Enable MDAP with custom k
            let config = MdapConfig {
                k,
                ..Default::default()
            };
            self.mdap_config = Some(config);
            self.set_status(
                LogLevel::Info,
                format!("Ready - Model: {} [MDAP] (Ctrl+C to quit)", self.model),
            );
            self.add_console_message(format!("✅ MDAP mode enabled with k={}", k));
        }
        self.clear_input();
    }

    /// Handle /mdap:target
    fn handle_mdap_set_target(&mut self, target: f64) {
        use crate::mdap::MdapConfig;

        if let Some(ref mut config) = self.mdap_config {
            config.target_success_rate = target;
            self.add_console_message(format!(
                "✅ MDAP target success rate set to {:.1}%",
                target * 100.0
            ));
        } else {
            // Enable MDAP with custom target
            let config = MdapConfig {
                target_success_rate: target,
                ..Default::default()
            };
            self.mdap_config = Some(config);
            self.set_status(
                LogLevel::Info,
                format!("Ready - Model: {} [MDAP] (Ctrl+C to quit)", self.model),
            );
            self.add_console_message(format!(
                "✅ MDAP mode enabled with target={:.1}%",
                target * 100.0
            ));
        }
        self.clear_input();
    }

    /// Handle /dream:run — run one consolidation cycle against the active
    /// in-memory conversation. The summarised messages are *not* written back
    /// to the live TUI buffer; this is strictly a report path until the
    /// persistence-aware adapter lands.
    async fn handle_dream_run(&mut self) {
        use brainwires::core::{Message, MessageContent, Role};

        if self.messages.is_empty() {
            self.add_console_message(
                "ℹ️  Dream: no messages in the active session yet — nothing to consolidate."
                    .to_string(),
            );
            self.clear_input();
            return;
        }

        let session_key = format!("tui-session-{}", chrono::Utc::now().timestamp());
        let messages: Vec<Message> = self
            .messages
            .iter()
            .map(|m| Message {
                role: match m.role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "system" => Role::System,
                    _ => Role::User,
                },
                content: MessageContent::Text(m.content.clone()),
                name: None,
                metadata: None,
            })
            .collect();

        self.add_console_message("💤 Dream cycle starting...".to_string());
        let provider = self.provider.clone();
        match crate::dream::run_once(provider, session_key, messages).await {
            Ok((report, _after)) => {
                let body = crate::dream::format_report(&report);
                self.messages.push(TuiMessage {
                    role: "system".to_string(),
                    content: body,
                    created_at: chrono::Utc::now().timestamp(),
                });
            }
            Err(e) => {
                self.add_console_message(format!("❌ Dream cycle failed: {e}"));
            }
        }
        self.clear_input();
    }

    /// Handle /dream or /dream:status — show the last-cycle report.
    fn handle_dream_status(&mut self) {
        let msg = match crate::dream::last_report() {
            Some(report) => {
                format!(
                    "Last dream cycle:\n\n{}\nRun `/dream:run` to execute another cycle.",
                    crate::dream::format_report(&report)
                )
            }
            None => "Dream has not run yet this session. Use `/dream:run` to execute a consolidation cycle."
                .to_string(),
        };
        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: msg,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /context:add command
    fn handle_context_add(&mut self, path: &str, pinned: bool) {
        use crate::types::working_set::estimate_tokens;
        use std::path::PathBuf;

        // Resolve path
        let file_path = if PathBuf::from(path).is_absolute() {
            PathBuf::from(path)
        } else {
            PathBuf::from(&self.working_directory).join(path)
        };

        // Try to canonicalize
        let file_path = file_path.canonicalize().unwrap_or(file_path);

        if !file_path.exists() {
            self.add_console_message(format!("❌ File not found: {}", file_path.display()));
            return;
        }

        // Read file to estimate tokens
        match std::fs::read_to_string(&file_path) {
            Ok(content) => {
                let tokens = estimate_tokens(&content);
                let file_name = file_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.to_string());

                if pinned {
                    self.working_set
                        .add_pinned(file_path.clone(), tokens, Some(&file_name));
                    self.add_console_message(format!(
                        "📌 Added and pinned: {} (~{} tokens)",
                        file_path.display(),
                        tokens
                    ));
                } else {
                    let eviction = self.working_set.add(file_path.clone(), tokens);
                    self.add_console_message(format!(
                        "✅ Added: {} (~{} tokens)",
                        file_path.display(),
                        tokens
                    ));
                    if let Some(reason) = eviction {
                        self.add_console_message(format!("⚠️  {}", reason));
                    }
                }
            }
            Err(e) => {
                self.add_console_message(format!("❌ Failed to read file: {}", e));
            }
        }
    }

    /// Handle /context:remove command
    fn handle_context_remove(&mut self, path: &str) {
        use std::path::PathBuf;

        let file_path = if PathBuf::from(path).is_absolute() {
            PathBuf::from(path)
        } else {
            PathBuf::from(&self.working_directory).join(path)
        };
        let file_path = file_path.canonicalize().unwrap_or(file_path);

        if self.working_set.remove(&file_path) {
            self.add_console_message(format!("✅ Removed: {}", file_path.display()));
        } else {
            self.add_console_message(format!("⚠️  Not in working set: {}", file_path.display()));
        }
    }

    /// Handle /context:pin command
    fn handle_context_pin(&mut self, path: &str) {
        use std::path::PathBuf;

        let file_path = if PathBuf::from(path).is_absolute() {
            PathBuf::from(path)
        } else {
            PathBuf::from(&self.working_directory).join(path)
        };
        let file_path = file_path.canonicalize().unwrap_or(file_path);

        if self.working_set.pin(&file_path) {
            self.add_console_message(format!("📌 Pinned: {}", file_path.display()));
        } else {
            self.add_console_message(format!(
                "⚠️  Not in working set: {}. Add it first with /context:add",
                file_path.display()
            ));
        }
    }

    /// Handle /context:unpin command
    fn handle_context_unpin(&mut self, path: &str) {
        use std::path::PathBuf;

        let file_path = if PathBuf::from(path).is_absolute() {
            PathBuf::from(path)
        } else {
            PathBuf::from(&self.working_directory).join(path)
        };
        let file_path = file_path.canonicalize().unwrap_or(file_path);

        if self.working_set.unpin(&file_path) {
            self.add_console_message(format!("✅ Unpinned: {}", file_path.display()));
        } else {
            self.add_console_message(format!("⚠️  Not in working set: {}", file_path.display()));
        }
    }

    /// Handle /context:clear command
    fn handle_context_clear(&mut self, keep_pinned: bool) {
        let count_before = self.working_set.len();
        self.working_set.clear(keep_pinned);
        let count_after = self.working_set.len();
        let removed = count_before - count_after;

        if keep_pinned && count_after > 0 {
            self.add_console_message(format!(
                "✅ Cleared {} file(s), kept {} pinned",
                removed, count_after
            ));
        } else {
            self.add_console_message(format!("✅ Cleared {} file(s) from working set", removed));
        }
    }

    /// Handle /tools (show current mode)
    fn handle_show_tool_mode(&mut self) {
        use crate::types::tool::ToolMode;

        let registry = brainwires_tool_builtins::registry_with_builtins();
        let builtin_count = registry.get_all().len();
        let mcp_count = self.mcp_tools.len();
        let total = builtin_count + mcp_count;

        let mode_str = match &self.tool_mode {
            ToolMode::Full => format!("full ({} built-in + {} MCP)", builtin_count, mcp_count),
            ToolMode::Explicit(tools) => {
                let builtin = tools.iter().filter(|t| !t.starts_with("mcp_")).count();
                let mcp = tools.iter().filter(|t| t.starts_with("mcp_")).count();
                format!("explicit ({} built-in + {} MCP selected)", builtin, mcp)
            }
            ToolMode::Smart => "smart (auto-select based on query)".to_string(),
            ToolMode::Core => format!("core ({} essential tools)", self.tools.len()),
            ToolMode::None => "none (tools disabled)".to_string(),
        };

        let mcp_servers_str = if self.mcp_connected_servers.is_empty() {
            "none".to_string()
        } else {
            self.mcp_connected_servers.join(", ")
        };

        let msg = format!(
            "Tool Mode: {}\n\n\
            Available modes:\n\
            • /tools full     - All {} tools ({} built-in + {} MCP)\n\
            • /tools explicit - Pick specific tools\n\
            • /tools smart    - Auto-select based on query (default)\n\
            • /tools core     - Core {} tools only\n\
            • /tools none     - Disable all tools\n\n\
            Connected MCP servers: {}",
            mode_str,
            total,
            builtin_count,
            mcp_count,
            registry.get_core().len(),
            mcp_servers_str
        );

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: msg,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /tools <mode> (set tool mode)
    fn handle_set_tool_mode(&mut self, mode: crate::types::tool::ToolMode) {
        use crate::types::tool::ToolMode;

        let registry = brainwires_tool_builtins::registry_with_builtins();

        self.tools = match &mode {
            ToolMode::Full => {
                // Built-in + MCP tools
                registry.get_all_with_mcp(&self.mcp_tools)
            }
            ToolMode::Explicit(names) => {
                // Include both built-in and MCP tools by name
                let mut tools: Vec<_> = names
                    .iter()
                    .filter_map(|name| registry.get(name).cloned())
                    .collect();
                // Add MCP tools that match
                tools.extend(
                    self.mcp_tools
                        .iter()
                        .filter(|t| names.contains(&t.name))
                        .cloned(),
                );
                tools
            }
            ToolMode::Smart => registry.get_core().into_iter().cloned().collect(),
            ToolMode::Core => registry.get_core().into_iter().cloned().collect(),
            ToolMode::None => vec![],
        };

        let count = self.tools.len();
        let mode_name = mode.display_name();
        self.tool_mode = mode;

        self.set_status(
            LogLevel::Info,
            format!("Tool mode: {} ({} tools)", mode_name, count),
        );
        self.add_console_message(format!(
            "✅ Tool mode set to: {} ({} tools active)",
            mode_name, count
        ));
        self.clear_input();
    }

    /// Handle /tools explicit (open tool picker)
    fn handle_open_tool_picker(&mut self) {
        use crate::tools::ToolCategory;
        use crate::tui::app::state::ToolPickerState;
        use crate::types::tool::ToolMode;
        use std::collections::{HashMap, HashSet};

        let registry = brainwires_tool_builtins::registry_with_builtins();

        // Get currently selected tools (if already in explicit mode)
        let selected_names: HashSet<String> = match &self.tool_mode {
            ToolMode::Explicit(names) => names.iter().cloned().collect(),
            _ => HashSet::new(),
        };

        // Build categories with their tools
        let categories = vec![
            ("File Operations", ToolCategory::FileOps),
            ("Search", ToolCategory::Search),
            ("Semantic Search", ToolCategory::SemanticSearch),
            ("Git", ToolCategory::Git),
            ("Web Search", ToolCategory::WebSearch),
            ("Web/HTTP", ToolCategory::Web),
            ("Bash/Shell", ToolCategory::Bash),
            ("Task Manager", ToolCategory::TaskManager),
            ("Agent Pool", ToolCategory::AgentPool),
            ("Planning", ToolCategory::Planning),
            ("Context", ToolCategory::Context),
        ];

        let mut picker_categories = Vec::new();
        for (name, category) in categories {
            let tools: Vec<(String, String, bool)> = registry
                .get_by_category(category)
                .iter()
                .map(|t| {
                    (
                        t.name.clone(),
                        t.description.clone(),
                        selected_names.contains(&t.name),
                    )
                })
                .collect();
            if !tools.is_empty() {
                picker_categories.push((name.to_string(), tools));
            }
        }

        // Add MCP server categories
        // Group MCP tools by server name
        let mut mcp_by_server: HashMap<String, Vec<(String, String, bool)>> = HashMap::new();

        for tool in &self.mcp_tools {
            // Extract server name from mcp_{server}_{tool} format
            if let Some(server_name) = extract_mcp_server_name(&tool.name) {
                let is_selected = selected_names.contains(&tool.name);

                mcp_by_server.entry(server_name.clone()).or_default().push((
                    tool.name.clone(),
                    tool.description.clone(),
                    is_selected,
                ));
            }
        }

        // Add MCP servers as categories with "MCP: " prefix
        let mut server_names: Vec<_> = mcp_by_server.keys().cloned().collect();
        server_names.sort();
        for server_name in server_names {
            if let Some(tools) = mcp_by_server.remove(&server_name)
                && !tools.is_empty()
            {
                picker_categories.push((format!("MCP: {}", server_name), tools));
            }
        }

        self.tool_picker_state = Some(ToolPickerState {
            categories: picker_categories,
            selected_category: 0,
            selected_tool: None,
            scroll: 0,
            filter_query: String::new(),
            collapsed: HashSet::new(),
        });

        self.mode = AppMode::ToolPicker;
        self.set_status(
            LogLevel::Info,
            "Select tools (Space: toggle, Enter: confirm, A: all, N: none, Esc: cancel)",
        );
        self.clear_input();
    }

    /// Confirm tool selection and apply explicit mode
    pub fn confirm_tool_selection(&mut self) {
        use crate::types::tool::ToolMode;

        if let Some(state) = &self.tool_picker_state {
            let selected: Vec<String> = state
                .categories
                .iter()
                .flat_map(|(_, tools)| tools.iter())
                .filter(|(_, _, selected)| *selected)
                .map(|(name, _, _)| name.clone())
                .collect();

            let registry = brainwires_tool_builtins::registry_with_builtins();

            // Get built-in tools
            let mut tools: Vec<_> = selected
                .iter()
                .filter_map(|name| registry.get(name).cloned())
                .collect();

            // Add MCP tools that were selected
            tools.extend(
                self.mcp_tools
                    .iter()
                    .filter(|t| selected.contains(&t.name))
                    .cloned(),
            );

            let count = tools.len();
            self.tools = tools;
            self.tool_mode = ToolMode::Explicit(selected);

            self.set_status(
                LogLevel::Info,
                format!("Tool mode: explicit ({} tools selected)", count),
            );
            self.add_console_message(format!("✅ Selected {} tools", count));
        }

        self.tool_picker_state = None;
        self.mode = AppMode::Normal;
    }
}

// Split submodules — each defines further `impl App` blocks in its own file.
// The command_handler module is just a namespace for these topic-grouped
// handler impls; dispatch and the smaller mdap/context/tools groups stay in
// mod.rs for now.
mod agents;
mod knowledge;
mod profile;
mod skills;

/// Extract server name from an MCP tool name (`mcp_<server>_<tool>`).
/// Used only by the dispatch code in this file; `truncate_description`
/// stays private inside `skills.rs` since only skill listings use it.
fn extract_mcp_server_name(tool_name: &str) -> Option<String> {
    if tool_name.starts_with("mcp_") {
        let rest = tool_name.strip_prefix("mcp_")?;
        if let Some(idx) = rest.find('_') {
            return Some(rest[..idx].to_string());
        }
    }
    None
}
