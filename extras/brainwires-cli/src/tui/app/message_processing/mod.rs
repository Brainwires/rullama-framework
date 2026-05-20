//! Message Processing and AI Interaction
//!
//! Handles message submission, AI streaming, and command execution.

mod checkpoint_handlers;
mod command_handler;
mod plan_crud;
mod plan_execution;
mod plan_hierarchy;
mod plan_lifecycle;
mod task_handlers;
mod template_handlers;

use super::state::{App, AppMode, LogLevel, TuiMessage};
use crate::agents::OrchestratorAgent;
use crate::types::agent::PermissionMode;
use crate::types::message::{Message, MessageContent, Role, StreamChunk};
use crate::types::provider::ChatOptions;
use anyhow::Result;
use futures::StreamExt;
use tokio::sync::mpsc;

pub(super) trait MessageProcessing {
    fn submit_message(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + '_>>;
}

impl MessageProcessing for App {
    /// Submit the current input as a message
    fn submit_message(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + '_>> {
        Box::pin(async move { self.submit_message_impl().await })
    }
}

impl App {
    /// Implementation of submit_message
    fn submit_message_impl(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + '_>> {
        Box::pin(async move { self.process_message().await })
    }

    /// Process a single message (internal, non-recursive)
    async fn process_message(&mut self) -> Result<()> {
        let user_content = self.input_text();

        // Add to input history first (before any processing that might clear input)
        let _ = self.prompt_history.add(user_content.clone());

        // Track whether we should call AI after command processing
        let mut should_call_ai = true;

        // Check if input is a slash command
        if let Some((cmd_name, cmd_args)) = self.command_executor.parse_input(&user_content) {
            // handle_command returns true if we should skip AI (command was an action)
            should_call_ai = !self
                .handle_command(cmd_name, &cmd_args, user_content)
                .await?;
        } else {
            // Not a command, add user message to display
            let user_message = TuiMessage {
                role: "user".to_string(),
                content: user_content.clone(),
                created_at: chrono::Utc::now().timestamp(),
            };
            self.messages.push(user_message);

            // Add to conversation history
            self.conversation_history.push(Message {
                role: Role::User,
                content: MessageContent::Text(user_content.clone()),
                name: None,
                metadata: None,
            });

            // PKS: Process user message for implicit fact detection
            let detected_count = self.pks_integration.process_user_message(&user_content);
            if detected_count > 0 {
                tracing::debug!(
                    "PKS: Detected {} implicit facts from user message",
                    detected_count
                );
            }

            // SkillRouter suggestion: if the user's message looks like a
            // discovered skill's purpose, print a hint. Non-intrusive —
            // we never auto-invoke; the user still has to type `/<name>`.
            if let Some(hint) = self.suggest_skill_for(&user_content) {
                self.add_console_message(hint);
            }
        }

        // Only proceed with AI if we added a user message AND command didn't skip AI
        if should_call_ai && !self.conversation_history.is_empty() {
            // Clear input and draft
            self.clear_input();
            self.input_draft = None;

            // Call AI
            self.call_ai_provider().await?;
        }

        Ok(())
    }

    /// Start AI streaming in background task (non-blocking)
    /// Returns immediately after spawning the background task.
    /// The main loop should poll stream events using poll_stream_events().
    pub async fn call_ai_provider(&mut self) -> Result<()> {
        use super::state::StreamEvent;
        use tokio::sync::mpsc;

        // In IPC mode, route request to Session instead of calling provider directly
        if self.is_ipc_mode {
            return self.call_ai_provider_ipc().await;
        }

        // Check if MDAP mode is enabled - use synchronous execution path
        if let Some(ref mdap_config) = self.mdap_config {
            return self.call_ai_provider_mdap(mdap_config.clone()).await;
        }

        // Enter waiting mode
        self.mode = AppMode::Waiting;
        let status_msg = if self.active_plan.is_some() {
            "Working on plan...".to_string()
        } else {
            "Streaming response...".to_string()
        };
        self.set_status(LogLevel::Info, status_msg);

        // Clone user_content before calling AI (for saving to storage later)
        let user_content = self
            .conversation_history
            .last()
            .and_then(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // Apply SEAL preprocessing for coreference resolution
        let resolved_query = self.seal_preprocess(&user_content);

        // If SEAL resolved something different, log it for debugging
        if resolved_query != user_content && self.seal_status.show_status {
            self.add_console_message(format!(
                "SEAL: \"{}\" → \"{}\"",
                if user_content.len() > 50 {
                    format!("{}...", &user_content[..50])
                } else {
                    user_content.clone()
                },
                if resolved_query.len() > 50 {
                    format!("{}...", &resolved_query[..50])
                } else {
                    resolved_query.clone()
                }
            ));
        }

        // Build conversation with active plan context injected
        let mut conversation_clone = self.conversation_history.clone();

        // If there's an active plan, inject it as a system message at the start (with task progress)
        // Also inject question instructions during planning stage
        if let Some(plan_context) = self.get_active_plan_context_with_progress().await {
            // Insert plan context as first message (after any existing system messages)
            let plan_system_msg = Message {
                role: Role::System,
                content: MessageContent::Text(plan_context),
                name: None,
                metadata: None,
            };
            // Insert at position 0 or after existing system messages
            let insert_pos = conversation_clone
                .iter()
                .take_while(|m| m.role == Role::System)
                .count();
            conversation_clone.insert(insert_pos, plan_system_msg);

            // Also inject question instructions during planning
            let question_instructions =
                crate::utils::question_instructions::get_question_instructions();
            let question_system_msg = Message {
                role: Role::System,
                content: MessageContent::Text(question_instructions.to_string()),
                name: None,
                metadata: None,
            };
            let insert_pos = conversation_clone
                .iter()
                .take_while(|m| m.role == Role::System)
                .count();
            conversation_clone.insert(insert_pos, question_system_msg);
        }

        // Inject working set files as a system message if non-empty
        if let Some(working_set_context) = self.working_set.build_context_injection() {
            let ws_system_msg = Message {
                role: Role::System,
                content: MessageContent::Text(working_set_context),
                name: None,
                metadata: None,
            };
            // Insert after system messages but before conversation
            let insert_pos = conversation_clone
                .iter()
                .take_while(|m| m.role == Role::System)
                .count();
            conversation_clone.insert(insert_pos, ws_system_msg);

            // Increment turn counter for working set (tracks access freshness)
            self.working_set.next_turn();
        }

        // Extract system prompt from conversation history and pass it in ChatOptions
        let system_prompt = conversation_clone
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.text().map(|s| s.to_string()));

        // Add placeholder for assistant message
        let assistant_msg_idx = self.messages.len();
        self.messages.push(TuiMessage {
            role: "assistant".to_string(),
            content: String::new(),
            created_at: chrono::Utc::now().timestamp(),
        });

        // Save state for streaming
        self.streaming_content = String::new();
        self.streaming_msg_idx = Some(assistant_msg_idx);
        self.streaming_conversation = Some(conversation_clone.clone());
        self.streaming_user_content = Some(user_content);

        // Create channel for stream events
        let (tx, rx) = mpsc::unbounded_channel::<StreamEvent>();
        self.stream_rx = Some(rx);

        // Clone what we need for the background task
        let provider = self.provider.clone();

        // Get tools based on current tool mode
        let tools = match &self.tool_mode {
            crate::types::tool::ToolMode::Smart => {
                // Smart routing: analyze messages to determine needed tools
                crate::tools::get_smart_tools(
                    &conversation_clone,
                    &brainwires_tool_builtins::registry_with_builtins(),
                )
            }
            _ => self.tools.clone(),
        };

        // Apply prompt mode filtering (Ask mode removes write tools) and
        // any pending skill tool scope. `apply_and_clear_skill_tool_scope`
        // is a no-op when no skill invocation is in flight.
        let tools = self.filter_tools_for_prompt_mode(tools);
        let tools = self.apply_and_clear_skill_tool_scope(tools);

        let options = ChatOptions {
            system: system_prompt,
            ..ChatOptions::default()
        };

        // Create cancellation token for streaming operation
        let cancel_token = tokio_util::sync::CancellationToken::new();
        self.cancellation_token = Some(cancel_token.clone());

        // Spawn background task to process stream with cancellation support
        let stream_handle = tokio::spawn(async move {
            let mut stream = provider.stream_chat(&conversation_clone, Some(&tools), &options);

            loop {
                // Check for cancellation between chunks
                tokio::select! {
                    biased; // Check cancellation first
                    _ = cancel_token.cancelled() => {
                        // Cancelled - send error event and exit
                        let _ = tx.send(StreamEvent::Error("Operation cancelled by user".to_string()));
                        break;
                    }
                    chunk_opt = stream.next() => {
                        let chunk = match chunk_opt {
                            Some(c) => c,
                            None => {
                                // Stream ended
                                let _ = tx.send(StreamEvent::Done);
                                break;
                            }
                        };

                        let is_terminal = matches!(chunk, Ok(StreamChunk::Done) | Err(_));

                        let event = match chunk {
                            Ok(StreamChunk::Text(text)) => StreamEvent::Text(text),
                            Ok(StreamChunk::ToolCall {
                                call_id,
                                response_id,
                                chat_id,
                                tool_name,
                                server,
                                parameters,
                            }) => StreamEvent::ToolCall {
                                call_id,
                                response_id,
                                chat_id,
                                tool_name,
                                server,
                                parameters,
                            },
                            Ok(StreamChunk::Done) => StreamEvent::Done,
                            Ok(_) => continue, // Ignore other chunk types
                            Err(e) => StreamEvent::Error(e.to_string()),
                        };

                        // Send event to main loop - stop if receiver dropped
                        if tx.send(event).is_err() {
                            break;
                        }

                        // Stop after terminal events (Done or Error)
                        if is_terminal {
                            break;
                        }
                    }
                }
            }
        });

        // Store handle for potential abort
        self.stream_task_handle = Some(stream_handle);

        Ok(())
    }

    /// Call AI provider via IPC (for viewer mode)
    ///
    /// In IPC mode, we send the user input to the Session (Agent) and receive
    /// streaming responses via IPC events. The Session handles all AI/tools/MCP.
    async fn call_ai_provider_ipc(&mut self) -> Result<()> {
        use brainwires::agent_network::ipc::ViewerMessage;

        // The pending_skill_tool_scope lives on the client side; the remote
        // session owns its own ToolExecutor and tool list. We can't enforce
        // scope over the wire today — surface the limitation to the user
        // and clear the scope so it doesn't leak.
        if self.pending_skill_tool_scope.take().is_some() {
            self.add_console_message(
                "⚠️  Skill allowed_tools is not yet enforced over IPC — tool scope skipped"
                    .to_string(),
            );
        }

        // Get the last user message content
        let user_content = self
            .conversation_history
            .last()
            .and_then(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // Get working set files for context
        let context_files: Vec<String> = self
            .working_set
            .file_paths()
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        // Enter waiting mode and journal the status before borrowing ipc_writer
        self.mode = AppMode::Waiting;
        self.set_status(LogLevel::Info, "Sending to session...");

        // Add placeholder assistant message for streaming
        let assistant_msg = TuiMessage {
            role: "assistant".to_string(),
            content: String::new(),
            created_at: chrono::Utc::now().timestamp(),
        };
        self.messages.push(assistant_msg);
        self.streaming_msg_idx = Some(self.messages.len() - 1);
        self.streaming_content = String::new();

        // Get the IPC writer and send (borrow is scoped to the block)
        let msg = ViewerMessage::UserInput {
            content: user_content,
            context_files,
        };
        {
            let writer = self
                .ipc_writer
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("No IPC connection in IPC mode"))?;
            writer.write(&msg).await?;
        }

        self.set_status(LogLevel::Info, "Streaming from session...");
        self.add_console_message("📤 Sent input to session via IPC".to_string());

        // Streaming responses will be received via Event::Ipc in the main event loop
        Ok(())
    }

    /// Handle an IPC message from Session (event-driven)
    ///
    /// Called from the main event loop when Event::Ipc is received.
    /// This is the event-driven replacement for the old polling approach.
    pub fn handle_ipc_event(&mut self, msg: brainwires::agent_network::ipc::AgentMessage) {
        use brainwires::agent_network::ipc::AgentMessage;

        match msg {
            AgentMessage::StreamChunk { text } => {
                self.streaming_content.push_str(&text);
                if let Some(idx) = self.streaming_msg_idx
                    && let Some(msg) = self.messages.get_mut(idx)
                {
                    msg.content = self.streaming_content.clone();
                }
            }
            AgentMessage::StreamEnd { .. } => {
                self.set_status(LogLevel::Info, "Ready");
                self.transition_to_normal_after_streaming();
                self.streaming_msg_idx = None;
            }
            AgentMessage::ToolCallStart { name, .. } => {
                self.set_status(LogLevel::Info, format!("Tool: {} (executing...)", name));
            }
            AgentMessage::ToolResult {
                name,
                output,
                error,
                ..
            } => {
                let success = error.is_none();
                let icon = if success { "✅" } else { "❌" };
                self.add_console_message(format!("{} Tool: {}", icon, name));

                // Record tool execution for Journal display (IPC mode)
                self.record_tool_execution(
                    &name,
                    &serde_json::Value::Null, // Parameters not available in IPC mode
                    output.as_deref().or(error.as_deref()),
                    success,
                    None,
                );
            }
            AgentMessage::StatusUpdate { status } => {
                self.set_status(LogLevel::Info, status);
            }
            AgentMessage::Error { message, .. } => {
                self.set_status(LogLevel::Error, format!("Error: {}", message));
                self.transition_to_normal_after_streaming();
            }
            AgentMessage::ConversationSync {
                messages,
                status,
                tool_mode,
                mcp_servers,
                ..
            } => {
                self.messages = messages
                    .iter()
                    .map(|m| TuiMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                        created_at: m.created_at,
                    })
                    .collect();
                self.set_status(LogLevel::Info, status);
                self.tool_mode = tool_mode;
                self.mcp_connected_servers = mcp_servers;
            }
            AgentMessage::MessageAdded { message } => {
                // Check for duplicates to avoid echo:
                // - TUI types locally → adds message → agent broadcasts it back → skip
                // - Agent finishes streaming → broadcasts assistant message → TUI already has it → skip
                //
                // For user messages, we check recent messages (not just last) because there
                // might be an assistant placeholder after the user message.
                if message.role == "user" {
                    // Check last 3 messages for duplicate user content
                    let already_exists = self
                        .messages
                        .iter()
                        .rev()
                        .take(3)
                        .any(|m| m.role == "user" && m.content == message.content);
                    if already_exists {
                        return;
                    }
                } else if message.role == "assistant" {
                    // For assistant messages, check if the last message is an assistant
                    // (streaming already populated it)
                    if let Some(last) = self.messages.last()
                        && last.role == "assistant"
                    {
                        return;
                    }
                }

                // Add new message (e.g., user input from GUI or assistant response)
                self.messages.push(TuiMessage {
                    role: message.role.clone(),
                    content: message.content.clone(),
                    created_at: message.created_at,
                });

                // If it's a user message, we might be about to receive a stream
                if message.role == "user" {
                    self.set_status(LogLevel::Info, "Working...");
                    self.mode = AppMode::Waiting;
                    // Prepare for assistant response
                    self.streaming_content.clear();
                    self.messages.push(TuiMessage {
                        role: "assistant".to_string(),
                        content: String::new(),
                        created_at: chrono::Utc::now().timestamp(),
                    });
                    self.streaming_msg_idx = Some(self.messages.len() - 1);
                }
            }
            AgentMessage::Exiting { reason } => {
                self.add_console_message(format!("⚠️ Session exiting: {}", reason));
                self.set_status(LogLevel::Warn, "Session ended");
                self.ipc_needs_respawn = true;
            }
            _ => {}
        }
    }

    /// Respawn the Session and reconnect
    ///
    /// Called when the Session connection is lost. Spawns a new Session
    /// and reconnects, preserving state from storage.
    ///
    /// Returns the new IpcReader so the caller can restart the IPC reader task.
    pub async fn respawn_session(&mut self) -> Result<brainwires::agent_network::ipc::IpcReader> {
        use crate::agent::spawn::spawn_agent_process;
        use brainwires::agent_network::ipc::{AgentMessage, Handshake};

        self.add_console_message("🔄 Respawning session...".to_string());
        self.set_status(LogLevel::Warn, "Respawning session...");

        // Spawn a new Session process
        let socket_path = spawn_agent_process(
            &self.session_id,
            Some(&self.model),
            None, // No MDAP config for respawn
        )
        .await?;

        self.add_console_message(format!(
            "✅ Session respawned at: {}",
            socket_path.display()
        ));

        // Connect to the new Session
        let mut conn = crate::ipc::connect_to_agent(&self.session_id).await?;

        // Send handshake
        let handshake = Handshake::new_session();
        conn.writer.write(&handshake).await?;

        // Wait for handshake response
        use brainwires::agent_network::ipc::HandshakeResponse;
        let response: HandshakeResponse = conn
            .reader
            .read()
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session closed during handshake"))?;

        if !response.accepted {
            anyhow::bail!(
                "Session rejected reconnection: {}",
                response.error.unwrap_or_default()
            );
        }

        // Wait for ConversationSync to restore state
        let sync: AgentMessage = conn
            .reader
            .read()
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session closed before sync"))?;

        if let AgentMessage::ConversationSync {
            messages,
            status,
            model,
            tool_mode,
            mcp_servers,
            ..
        } = sync
        {
            // Restore state from Session
            self.messages = messages
                .iter()
                .map(|m| TuiMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    created_at: m.created_at,
                })
                .collect();
            self.set_status(LogLevel::Info, status);
            self.model = model;
            self.tool_mode = tool_mode;
            self.mcp_connected_servers = mcp_servers;
        }

        // Split the connection - store writer, return reader
        let (reader, writer) = conn.split();
        self.ipc_writer = Some(writer);
        self.mode = AppMode::Normal;
        self.ipc_needs_respawn = false;

        self.add_console_message("✅ Reconnected to session".to_string());

        Ok(reader)
    }

    /// Poll for stream events and process them (non-blocking)
    /// Returns true if streaming is still active, false if done
    /// This should be called from the main event loop after each render
    pub async fn poll_stream_events(&mut self) -> bool {
        use super::state::{PendingToolData, StreamEvent};

        // Take the receiver temporarily to avoid borrow issues
        let mut rx = match self.stream_rx.take() {
            Some(rx) => rx,
            None => return false, // No active stream
        };

        // Collect events first, then process them
        let mut events: Vec<StreamEvent> = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(event) => events.push(event),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    events.push(StreamEvent::Done);
                    break;
                }
            }
        }

        // Process collected events
        let mut pending_tool_call: Option<(
            String,
            String,
            Option<String>,
            String,
            String,
            serde_json::Value,
        )> = None;
        let mut stream_done = false;
        let mut console_messages: Vec<String> = Vec::new();

        for event in events {
            match event {
                StreamEvent::Text(text) => {
                    // Debug: log each text chunk received
                    // console_messages.push(format!("📝 Text chunk: {} bytes", text.len()));
                    self.streaming_content.push_str(&text);
                    // Update the assistant message
                    if let Some(idx) = self.streaming_msg_idx
                        && let Some(msg) = self.messages.get_mut(idx)
                    {
                        msg.content = self.streaming_content.clone();
                    }
                    // Note: scroll_to_bottom() is called in main loop AFTER draw()
                    // so that conversation_area is correctly set
                }
                StreamEvent::ToolCall {
                    call_id,
                    response_id,
                    chat_id,
                    tool_name,
                    server,
                    parameters,
                } => {
                    if server == "cli-local" {
                        self.set_status(
                            LogLevel::Info,
                            format!("Tool: {} (executing...)", tool_name),
                        );
                        console_messages.push(format!("🔧 Tool requested: {}", tool_name));
                        pending_tool_call =
                            Some((call_id, response_id, chat_id, tool_name, server, parameters));
                    } else {
                        console_messages
                            .push(format!("⚠️ Ignoring tool from unknown server: {}", server));
                    }
                }
                StreamEvent::Progress {
                    tool_name,
                    message,
                    progress,
                } => {
                    // Show progress update in status bar only (avoid spamming console)
                    let progress_str = progress
                        .map(|p| format!(" ({:.0}%)", p * 100.0))
                        .unwrap_or_default();
                    // Progress is ephemeral — set status directly to avoid journal spam
                    self.status = format!("⏳ {} - {}{}", tool_name, message, progress_str);
                }
                StreamEvent::Done => {
                    stream_done = true;
                }
                StreamEvent::Error(e) => {
                    self.set_status(LogLevel::Error, format!("Error: {}", e));
                    console_messages.push(format!("❌ Stream error: {}", e));
                    stream_done = true;
                }
                // ToolResult is handled by poll_tool_events, not here
                StreamEvent::ToolResult { .. } => {}
            }
        }

        // Add collected console messages
        for msg in console_messages {
            self.add_console_message(msg);
        }

        // Put back the receiver if not done
        if !stream_done {
            self.stream_rx = Some(rx);
        }

        // Spawn tool execution in background if one was requested
        if let Some((call_id, response_id, chat_id, tool_name, _server, parameters)) =
            pending_tool_call
        {
            self.add_console_message(format!(
                "🔧 Spawning background tool: {} (call_id: {})",
                tool_name,
                &call_id[..8.min(call_id.len())]
            ));

            // Get conversation clone for tool handler
            let conversation_clone = self.streaming_conversation.clone().unwrap_or_default();

            // Store pending tool data for continuation
            self.pending_tool_data = Some(PendingToolData {
                call_id: call_id.clone(),
                response_id: response_id.clone(),
                chat_id: chat_id.clone(),
                tool_name: tool_name.clone(),
                parameters: parameters.clone(),
                conversation: conversation_clone,
            });

            // Create channel for tool execution events
            let (tx, tool_rx) = mpsc::unbounded_channel::<StreamEvent>();
            self.tool_rx = Some(tool_rx);
            self.tool_tx = Some(tx.clone());

            // Clone what we need for the background task
            let tool_use = crate::types::tool::ToolUse {
                id: call_id.clone(),
                name: tool_name.clone(),
                input: parameters.clone(),
            };
            let tool_context = crate::types::tool::ToolContext {
                working_directory: self.working_directory.clone(),
                // Use full_access for TUI mode - users expect agents to have write access
                capabilities: serde_json::to_value(
                    brainwires::permissions::AgentCapabilities::full_access(),
                )
                .ok(),
                ..Default::default()
            };
            let tool_executor = self.tool_executor.clone();
            let tool_name_clone = tool_name.clone();

            // Create cancellation token for this operation
            let cancel_token = tokio_util::sync::CancellationToken::new();
            self.cancellation_token = Some(cancel_token.clone());

            // Check if this is an MCP tool (for progress notification support)
            let is_mcp_tool = tool_name.starts_with("mcp_");
            let mcp_tool_input = parameters.clone();

            // Spawn background task for tool execution with progress updates and cancellation
            let tool_handle = tokio::spawn(async move {
                // Send initial progress
                let _ = tx.send(StreamEvent::Progress {
                    tool_name: tool_name_clone.clone(),
                    message: "Starting execution...".to_string(),
                    progress: Some(0.0),
                });

                // For MCP tools, use the progress-aware execution path
                if is_mcp_tool {
                    // Create a channel for MCP progress notifications
                    let mcp_progress_tx = tx.clone();
                    let mcp_tool_name = tool_name_clone.clone();

                    // Create progress callback that forwards MCP progress to StreamEvent
                    let progress_callback = move |params: crate::mcp::ProgressParams| {
                        let progress_pct = if let Some(total) = params.total {
                            if total > 0.0 {
                                Some(params.progress / total)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let message = params.message.unwrap_or_else(|| {
                            if let Some(pct) = progress_pct {
                                format!("Progress: {:.0}%", pct * 100.0)
                            } else {
                                format!("Progress: {}", params.progress)
                            }
                        });

                        let _ = mcp_progress_tx.send(StreamEvent::Progress {
                            tool_name: mcp_tool_name.clone(),
                            message,
                            progress: progress_pct,
                        });
                    };

                    // Execute MCP tool with real progress notifications
                    tokio::select! {
                        _ = cancel_token.cancelled() => {
                            let _ = tx.send(StreamEvent::ToolResult {
                                tool_name: tool_name_clone,
                                result: None,
                                error: Some("Operation cancelled by user".to_string()),
                            });
                        }
                        result = crate::tools::McpToolExecutor::execute_with_progress(
                            &tool_use.name,
                            &mcp_tool_input,
                            progress_callback
                        ) => {
                            match result {
                                Ok(content) => {
                                    let _ = tx.send(StreamEvent::ToolResult {
                                        tool_name: tool_name_clone,
                                        result: Some(content),
                                        error: None,
                                    });
                                }
                                Err(e) => {
                                    let _ = tx.send(StreamEvent::ToolResult {
                                        tool_name: tool_name_clone,
                                        result: None,
                                        error: Some(e.to_string()),
                                    });
                                }
                            }
                        }
                    }
                } else {
                    // Non-MCP tools: use standard execution with synthetic progress
                    let progress_tx = tx.clone();
                    let progress_tool_name = tool_name_clone.clone();
                    let progress_cancel = cancel_token.clone();
                    let progress_handle = tokio::spawn(async move {
                        let mut elapsed_secs = 0u64;
                        loop {
                            tokio::select! {
                                _ = progress_cancel.cancelled() => {
                                    break; // Cancelled, stop progress updates
                                }
                                _ = tokio::time::sleep(tokio::time::Duration::from_secs(2)) => {
                                    elapsed_secs += 2;
                                    let message = match elapsed_secs {
                                        2..=5 => "Working...".to_string(),
                                        6..=15 => format!("Still working... ({}s)", elapsed_secs),
                                        16..=60 => format!("Processing... ({}s)", elapsed_secs),
                                        _ => format!("Long operation in progress... ({}s)", elapsed_secs),
                                    };
                                    if progress_tx.send(StreamEvent::Progress {
                                        tool_name: progress_tool_name.clone(),
                                        message,
                                        progress: None,
                                    }).is_err() {
                                        break; // Channel closed, stop updates
                                    }
                                }
                            }
                        }
                    });

                    // Execute tool with cancellation support
                    tokio::select! {
                        _ = cancel_token.cancelled() => {
                            // Cancelled - send cancellation result
                            progress_handle.abort();
                            let _ = tx.send(StreamEvent::ToolResult {
                                tool_name: tool_name_clone,
                                result: None,
                                error: Some("Operation cancelled by user".to_string()),
                            });
                        }
                        result = tool_executor.execute(&tool_use, &tool_context) => {
                            // Tool completed - cancel progress updates
                            progress_handle.abort();

                            match result {
                                Ok(tool_result) => {
                                    // If tool reported an error, include the content as the error message
                                    let error_msg = if tool_result.is_error {
                                        // Extract error details from content
                                        let error_detail = if tool_result.content.len() > 200 {
                                            format!("{}...", &tool_result.content[..200])
                                        } else {
                                            tool_result.content.clone()
                                        };
                                        Some(format!("Tool error: {}", error_detail))
                                    } else {
                                        None
                                    };

                                    let _ = tx.send(StreamEvent::ToolResult {
                                        tool_name: tool_name_clone,
                                        result: Some(tool_result.content),
                                        error: error_msg,
                                    });
                                }
                                Err(e) => {
                                    let _ = tx.send(StreamEvent::ToolResult {
                                        tool_name: tool_name_clone,
                                        result: None,
                                        error: Some(format!("Tool execution failed: {}", e)),
                                    });
                                }
                            }
                        }
                    }
                }
            });

            // Store handle for potential abort
            self.tool_task_handle = Some(tool_handle);

            // Don't finalize stream - wait for tool to complete
            return true;
        }

        // If stream is done, finalize
        if stream_done {
            self.finalize_stream().await;
            return false;
        }

        true // Still streaming
    }

    /// Poll for tool execution events (non-blocking)
    /// Returns true if tool execution is still active, false if done
    pub async fn poll_tool_events(&mut self) -> bool {
        use super::state::StreamEvent;

        // Take the receiver temporarily
        let mut rx = match self.tool_rx.take() {
            Some(rx) => rx,
            None => return false, // No active tool execution
        };

        // Collect events
        let mut events: Vec<StreamEvent> = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(event) => events.push(event),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        let mut tool_done = false;
        let mut tool_result: Option<String> = None;
        let mut tool_error: Option<String> = None;
        let mut tool_name_result: Option<String> = None;

        for event in events {
            match event {
                StreamEvent::Progress {
                    tool_name,
                    message,
                    progress,
                } => {
                    // Show progress update in status bar only (avoid spamming console)
                    let progress_str = progress
                        .map(|p| format!(" ({:.0}%)", p * 100.0))
                        .unwrap_or_default();
                    // Progress is ephemeral — set status directly to avoid journal spam
                    self.status = format!("⏳ {} - {}{}", tool_name, message, progress_str);
                }
                StreamEvent::ToolResult {
                    tool_name,
                    result,
                    error,
                } => {
                    tool_done = true;
                    tool_name_result = Some(tool_name);
                    tool_result = result;
                    tool_error = error;
                }
                _ => {}
            }
        }

        // Put back if not done
        if !tool_done {
            self.tool_rx = Some(rx);
            return true; // Still executing
        }

        // Tool execution completed - handle result and spawn continuation in background
        if let Some(pending) = self.pending_tool_data.take() {
            let tool_name = tool_name_result.unwrap_or(pending.tool_name.clone());

            if let Some(ref error) = tool_error {
                self.add_console_message(format!("❌ Tool {} failed: {}", tool_name, error));
                self.set_status(LogLevel::Error, format!("Tool error: {}", error));

                // Record failed tool execution for Journal display
                self.record_tool_execution(
                    &tool_name,
                    &pending.parameters,
                    Some(error),
                    false,
                    None,
                );

                // Clean up and finalize on error
                self.tool_rx = None;
                self.tool_tx = None;
                self.finalize_stream().await;
                return false;
            }

            if let Some(result) = tool_result {
                // Limit tool output to prevent context window overflow
                const MAX_TOOL_OUTPUT_CHARS: usize = 10_000;
                let truncated_output = if result.len() > MAX_TOOL_OUTPUT_CHARS {
                    let truncated = &result[..MAX_TOOL_OUTPUT_CHARS];
                    format!(
                        "{}\n\n[Output truncated: {} of {} characters]",
                        truncated,
                        MAX_TOOL_OUTPUT_CHARS,
                        result.len()
                    )
                } else {
                    result.clone()
                };

                let preview = if truncated_output.len() > 200 {
                    format!("{}...", &truncated_output[..200])
                } else {
                    truncated_output.clone()
                };
                self.add_console_message(format!("✅ Tool {} completed: {}", tool_name, preview));

                // Record successful tool execution for Journal display
                self.record_tool_execution(
                    &tool_name,
                    &pending.parameters,
                    Some(&truncated_output),
                    true,
                    None, // TODO: Track duration
                );

                // Spawn continuation request in background (non-blocking)
                self.set_status(LogLevel::Info, "Processing tool result...");
                self.spawn_continuation_request(pending, truncated_output);

                // Return true - continuation is now running, poll_stream_events will handle it
                return true;
            }
        }

        // Clean up tool execution state (only reached if no result/error)
        self.tool_rx = None;
        self.tool_tx = None;

        // Finalize the stream
        self.finalize_stream().await;

        false // Tool execution complete
    }

    /// Spawn continuation request in a background task (non-blocking)
    fn spawn_continuation_request(
        &mut self,
        pending: super::state::PendingToolData,
        tool_output: String,
    ) {
        use super::state::StreamEvent;
        use crate::cli::chat::continuation::{LogCallback, send_continuation_request};
        use std::sync::Arc;

        // Create channel for streaming the continuation response
        let (tx, rx) = mpsc::unbounded_channel::<StreamEvent>();
        self.stream_rx = Some(rx);

        // Clone what we need for the background task
        let provider = self.provider.clone();
        let model = self.model.clone();
        let tools = self.tools.clone();
        let working_directory = self.working_directory.clone();
        let conversation = pending.conversation.clone();

        // Spawn background task for continuation
        let stream_handle = tokio::spawn(async move {
            // Build agent context for continuation
            let agent_context = crate::types::agent::AgentContext {
                working_directory,
                user_id: None,
                conversation_history: conversation,
                tools,
                metadata: std::collections::HashMap::new(),
                working_set: crate::types::WorkingSet::new(),
                // Use full_access for TUI mode - users expect agents to have write access
                capabilities: brainwires::permissions::AgentCapabilities::full_access(),
            };

            // Create a no-op logger (logs go to console via event)
            let tui_logger: LogCallback = Arc::new(|_msg: &str| {});

            // Send continuation request
            let result = send_continuation_request(
                &provider,
                &agent_context,
                &model,
                pending.chat_id,
                &pending.response_id,
                &pending.call_id,
                &pending.tool_name,
                &pending.parameters,
                &tool_output,
                &[],
                tui_logger,
            )
            .await;

            // Send result back via channel
            match result {
                Ok(continuation_text) => {
                    // Send the continuation text as streaming content
                    let _ = tx.send(StreamEvent::Text(continuation_text));
                    let _ = tx.send(StreamEvent::Done);
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!("Continuation failed: {}", e)));
                }
            }
        });

        // Store handle for potential abort
        self.stream_task_handle = Some(stream_handle);

        // Clean up tool state (continuation is now a stream)
        self.tool_rx = None;
        self.tool_tx = None;
    }

    /// Finalize the stream after it completes
    async fn finalize_stream(&mut self) {
        use crate::tui::question_parser;
        use crate::types::question::QuestionAnswerState;

        // Clone streaming_content to avoid borrow issues
        let response_content = self.streaming_content.clone();

        // Parse response for clarifying questions
        let parsed = question_parser::parse_response(&response_content);
        let clean_content = parsed.content;
        let questions = parsed.questions;

        // Update the displayed message with clean content (question block removed)
        if let Some(idx) = self.streaming_msg_idx
            && let Some(msg) = self.messages.get_mut(idx)
        {
            msg.content = clean_content.clone();
        }

        // Add to conversation history (using clean content without question block)
        self.conversation_history.push(Message {
            role: Role::Assistant,
            content: MessageContent::Text(clean_content.clone()),
            name: None,
            metadata: None,
        });

        // Detect and auto-complete tasks from AI response
        if self.active_plan.is_some() {
            self.detect_and_complete_tasks(&clean_content).await;
        }

        // Save to storage
        let user_content = self.streaming_user_content.take().unwrap_or_default();
        self.save_conversation_to_storage(&user_content, &clean_content)
            .await;

        // Clean up streaming state
        self.stream_rx = None;
        self.streaming_content = String::new();
        self.streaming_msg_idx = None;
        self.streaming_conversation = None;

        // Check if questions were found in the response
        if let Some(question_block) = questions {
            // Initialize question state and enter question mode
            self.pending_questions = Some(question_block.clone());
            self.question_state = QuestionAnswerState::new(&question_block);
            self.mode = AppMode::QuestionAnswer;
            self.set_status(
                LogLevel::Info,
                format!(
                    "Clarifying Questions - {} question(s)",
                    question_block.questions.len()
                ),
            );
            return; // Don't return to normal mode yet
        }

        // Return to normal mode (preserves dialog modes if open)
        self.transition_to_normal_after_streaming();

        // Check if there are queued messages to process
        if !self.queued_messages.is_empty() {
            self.set_status(
                LogLevel::Info,
                format!("Ready - {} queued messages", self.queued_messages.len()),
            );
        } else {
            let mdap_indicator = if self.mdap_config.is_some() {
                " [MDAP]"
            } else {
                ""
            };
            self.set_status(
                LogLevel::Info,
                format!(
                    "Ready - Model: {}{} (Ctrl+C to quit)",
                    self.model, mdap_indicator
                ),
            );
        }
    }

    /// Call AI provider using MDAP (high-reliability mode)
    ///
    /// This uses the OrchestratorAgent with MDAP voting for increased reliability.
    /// Unlike the streaming path, MDAP execution is synchronous from the TUI's perspective
    /// since it requires voting consensus across multiple samples.
    async fn call_ai_provider_mdap(&mut self, mdap_config: crate::mdap::MdapConfig) -> Result<()> {
        use crate::types::agent::AgentContext;

        // Enter waiting mode with MDAP indicator
        self.mode = AppMode::Waiting;
        self.set_status(
            LogLevel::Info,
            format!("MDAP Processing (k={})...", mdap_config.k),
        );

        // Clone user_content before calling AI (for saving to storage later)
        let user_content = self
            .conversation_history
            .last()
            .and_then(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // Build conversation with active plan context injected
        let mut conversation_clone = self.conversation_history.clone();

        // If there's an active plan, inject it as a system message at the start (with task progress)
        // Also inject question instructions during planning stage
        if let Some(plan_context) = self.get_active_plan_context_with_progress().await {
            let plan_system_msg = Message {
                role: Role::System,
                content: MessageContent::Text(plan_context),
                name: None,
                metadata: None,
            };
            let insert_pos = conversation_clone
                .iter()
                .take_while(|m| m.role == Role::System)
                .count();
            conversation_clone.insert(insert_pos, plan_system_msg);

            // Also inject question instructions during planning
            let question_instructions =
                crate::utils::question_instructions::get_question_instructions();
            let question_system_msg = Message {
                role: Role::System,
                content: MessageContent::Text(question_instructions.to_string()),
                name: None,
                metadata: None,
            };
            let insert_pos = conversation_clone
                .iter()
                .take_while(|m| m.role == Role::System)
                .count();
            conversation_clone.insert(insert_pos, question_system_msg);
        }

        // Inject working set files as a system message if non-empty
        if let Some(working_set_context) = self.working_set.build_context_injection() {
            let ws_system_msg = Message {
                role: Role::System,
                content: MessageContent::Text(working_set_context),
                name: None,
                metadata: None,
            };
            let insert_pos = conversation_clone
                .iter()
                .take_while(|m| m.role == Role::System)
                .count();
            conversation_clone.insert(insert_pos, ws_system_msg);
            self.working_set.next_turn();
        }

        // Add placeholder for assistant message
        let assistant_msg_idx = self.messages.len();
        self.messages.push(TuiMessage {
            role: "assistant".to_string(),
            content: "Processing with MDAP...".to_string(),
            created_at: chrono::Utc::now().timestamp(),
        });

        // Build agent context for MDAP execution. Apply any pending
        // skill tool scope (consumed + cleared here so subsequent turns
        // see the full set).
        let scoped_tools = self.apply_and_clear_skill_tool_scope(self.tools.clone());
        let mut agent_context = AgentContext {
            working_directory: self.working_directory.clone(),
            user_id: None,
            conversation_history: conversation_clone,
            tools: scoped_tools,
            metadata: std::collections::HashMap::new(),
            working_set: crate::types::WorkingSet::new(),
            // Use full_access for TUI mode - users expect agents to have write access
            capabilities: brainwires::permissions::AgentCapabilities::full_access(),
        };

        // Create orchestrator and execute with MDAP
        let mut orchestrator = OrchestratorAgent::new(self.provider.clone(), PermissionMode::Auto);

        match orchestrator
            .execute_mdap(&user_content, &mut agent_context, mdap_config.clone())
            .await
        {
            Ok((response, metrics)) => {
                // Update the assistant message with the response
                if let Some(msg) = self.messages.get_mut(assistant_msg_idx) {
                    msg.content = response.message.clone();
                }

                // Add to conversation history
                self.conversation_history.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(response.message.clone()),
                    name: None,
                    metadata: None,
                });

                // Detect and auto-complete tasks from AI response
                if self.active_plan.is_some() {
                    self.detect_and_complete_tasks(&response.message).await;
                }

                // Save to storage
                self.save_conversation_to_storage(&user_content, &response.message)
                    .await;

                // Log MDAP metrics to console
                self.add_console_message(format!(
                    "✅ MDAP completed: {} steps, {} samples, {:.1}% red-flagged",
                    metrics.completed_steps,
                    metrics.total_samples,
                    (metrics.red_flagged_samples as f64 / metrics.total_samples.max(1) as f64)
                        * 100.0
                ));

                // Update status with success
                let mdap_indicator = if self.mdap_config.is_some() {
                    " [MDAP]"
                } else {
                    ""
                };
                self.set_status(
                    LogLevel::Info,
                    format!(
                        "Ready - Model: {}{} (Ctrl+C to quit)",
                        self.model, mdap_indicator
                    ),
                );
            }
            Err(e) => {
                // Update the assistant message with error
                if let Some(msg) = self.messages.get_mut(assistant_msg_idx) {
                    msg.content = format!("MDAP Error: {}", e);
                }

                self.add_console_message(format!("❌ MDAP error: {}", e));
                self.set_status(LogLevel::Error, format!("MDAP Error: {}", e));
            }
        }

        // Return to normal mode (preserves dialog modes if open)
        self.transition_to_normal_after_streaming();

        Ok(())
    }

    /// Save conversation to storage
    async fn save_conversation_to_storage(&mut self, user_content: &str, response_content: &str) {
        // Only save if we have content (not an empty response)
        if response_content.is_empty() {
            return;
        }

        let now = chrono::Utc::now().timestamp();

        // Track if this is the first message pair (for creating conversation)
        // conversation_history includes: system prompt + user message + assistant response
        // So after first complete exchange, len() == 3
        let is_first_pair = self.conversation_history.len() == 3;

        // Save the user message
        let user_msg = crate::storage::MessageMetadata {
            message_id: uuid::Uuid::new_v4().to_string(),
            conversation_id: self.session_id.clone(),
            role: "user".to_string(),
            content: user_content.to_string(),
            token_count: None,
            model_id: Some(self.model.clone()),
            images: None,
            created_at: now,
            expires_at: None,
        };
        if let Err(e) = self.message_store.add(user_msg).await {
            self.add_console_message(format!("Failed to save user message: {}", e));
        }

        // Save the assistant message
        let assistant_msg = crate::storage::MessageMetadata {
            message_id: uuid::Uuid::new_v4().to_string(),
            conversation_id: self.session_id.clone(),
            role: "assistant".to_string(),
            content: response_content.to_string(),
            token_count: None,
            model_id: Some(self.model.clone()),
            images: None,
            created_at: now,
            expires_at: None,
        };
        if let Err(e) = self.message_store.add(assistant_msg).await {
            self.add_console_message(format!("Failed to save assistant message: {}", e));
        }

        // Create or update conversation metadata
        if is_first_pair {
            // Create conversation with correct message count (messages are already saved)
            let title = if let Some(first_msg) = self.messages.first() {
                let content = &first_msg.content;
                if content.len() > 50 {
                    format!("{}...", &content[..47])
                } else {
                    content.clone()
                }
            } else {
                "New conversation".to_string()
            };

            // Pass message count to create so it's correct from the start
            let message_count = self.messages.len() as i32;
            if let Err(e) = self
                .conversation_store
                .create(
                    self.session_id.clone(),
                    Some(title),
                    Some(self.model.clone()),
                    Some(message_count),
                )
                .await
            {
                // Log error but don't fail the request
                self.add_console_message(format!("Failed to save conversation: {}", e));
            }
        } else {
            // Update existing conversation metadata (message count and updated_at timestamp)
            let message_count = self.messages.len() as i32;
            if let Err(e) = self
                .conversation_store
                .update(
                    &self.session_id,
                    None, // keep existing title
                    Some(message_count),
                )
                .await
            {
                // Only log if it's not a "not found" error
                if !e.to_string().contains("not found") {
                    self.add_console_message(format!("Failed to update conversation: {}", e));
                }
            }
        }
    }

    /// Check if there are queued messages and process the next one
    pub async fn process_queued_message(&mut self) -> Result<bool> {
        if self.queued_messages.is_empty() {
            return Ok(false);
        }

        // Take the first queued message
        let queued_message = self.queued_messages.remove(0);
        let remaining = self.queued_messages.len();

        // Set status to show we're processing queued message
        let queued_msg = if remaining > 0 {
            format!("Processing queued message ({} remaining)", remaining)
        } else {
            "Processing queued message".to_string()
        };
        self.set_status(LogLevel::Info, queued_msg);

        // Set the input to the queued message and submit it
        self.input_state.set_text(queued_message);

        // Submit the queued message
        self.submit_message().await?;

        Ok(true)
    }

    /// Update the config file to persist model selection
    fn update_config_model(model: &str) -> Result<()> {
        use crate::config::{ConfigManager, ConfigUpdates};

        let mut config_manager = ConfigManager::new()?;
        config_manager.update(ConfigUpdates {
            model: Some(model.to_string()),
            ..Default::default()
        });
        config_manager.save()?;

        Ok(())
    }

    /// Render the `/provider` list into a system message.
    pub(super) fn handle_list_providers(&mut self) {
        use super::state::TuiMessage;
        use crate::providers::ProviderType;
        use crate::types::provider_ext::{CHAT_PROVIDERS, summary};

        let current = match crate::config::ConfigManager::new() {
            Ok(mgr) => mgr.get().provider_type,
            Err(_) => ProviderType::Brainwires,
        };

        let mut lines = vec![format!("Current provider: {}", current.as_str())];
        lines.push(String::new());
        lines.push("Available providers (use `/provider <name>` to switch):".to_string());
        for p in CHAT_PROVIDERS {
            let marker = if *p == current { "*" } else { " " };
            lines.push(format!("  {} {:<14}  {}", marker, p.as_str(), summary(*p)));
        }

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: lines.join("\n"),
            created_at: chrono::Utc::now().timestamp(),
        });
    }

    /// Switch provider by updating config and rebuilding the Provider instance.
    ///
    /// Tolerates missing credentials — we only surface an error to the user;
    /// the session continues to use the previously active provider.
    pub(super) async fn handle_switch_provider(&mut self, name: String) {
        use super::state::{LogLevel, TuiMessage};
        use crate::config::{ConfigManager, ConfigUpdates};
        use crate::providers::{ProviderFactory, ProviderType};

        let target = match ProviderType::from_str_opt(&name) {
            Some(p) => p,
            None => {
                self.set_status(
                    LogLevel::Error,
                    format!(
                        "Unknown provider: '{}'. Try: /provider  to see the list.",
                        name
                    ),
                );
                return;
            }
        };

        // Persist the choice and update in-memory model to the provider's default.
        let new_model = target.default_model().to_string();
        let mut mgr = match ConfigManager::new() {
            Ok(m) => m,
            Err(e) => {
                self.set_status(LogLevel::Error, format!("Config load failed: {}", e));
                return;
            }
        };
        mgr.update(ConfigUpdates {
            provider_type: Some(target),
            model: Some(new_model.clone()),
            ..Default::default()
        });
        if let Err(e) = mgr.save() {
            self.set_status(LogLevel::Error, format!("Config save failed: {}", e));
            return;
        }

        // Rebuild the provider instance so subsequent turns use the new one.
        match ProviderFactory::new()
            .create_with_overrides(new_model.clone(), Some(target), None)
            .await
        {
            Ok(new_provider) => {
                self.provider = new_provider;
                self.model = new_model.clone();
                self.set_status(
                    LogLevel::Info,
                    format!("Switched to {} ({})", target.as_str(), new_model),
                );
            }
            Err(e) => {
                // Don't revert config — the user clearly wants this provider;
                // surface the credential hint so they can fix it.
                self.messages.push(TuiMessage {
                    role: "system".to_string(),
                    content: format!(
                        "Provider set to {} but couldn't start it: {}",
                        target.as_str(),
                        e
                    ),
                    created_at: chrono::Utc::now().timestamp(),
                });
                self.set_status(
                    LogLevel::Error,
                    format!("{} not ready — see message above", target.as_str()),
                );
            }
        }
    }

    /// Detect completed tasks from AI response and auto-complete them
    async fn detect_and_complete_tasks(&mut self, response: &str) {
        use crate::types::agent::TaskStatus;
        use crate::utils::completion_detector::CompletionDetector;

        // Get active (in-progress) tasks for detection
        let active_tasks: Vec<crate::types::agent::Task> = {
            let task_mgr = self.task_manager.read().await;
            task_mgr.get_tasks_by_status(TaskStatus::InProgress).await
        };

        if active_tasks.is_empty() {
            return;
        }

        // Detect completions
        let matches = CompletionDetector::detect_completed_tasks(response, &active_tasks);

        // Auto-complete detected tasks with high confidence
        let mut console_msgs = Vec::new();
        for m in matches {
            // Only auto-complete with confidence >= 0.8
            if m.confidence >= 0.8 {
                let summary = m.summary.unwrap_or_else(|| "Detected complete".to_string());
                let task_mgr = self.task_manager.write().await;
                if let Ok(()) = task_mgr.complete_task(&m.task_id, summary.clone()).await {
                    // Persist task state
                    if let Some(task) = task_mgr.get_task(&m.task_id).await {
                        let _ = self.task_store.save(&task, &self.session_id).await;
                    }

                    // Update status line to show task completion
                    if let Some(task) = active_tasks.iter().find(|t| t.id == m.task_id) {
                        console_msgs.push(format!(
                            "✓ Auto-completed: {} (confidence: {:.0}%)",
                            task.description,
                            m.confidence * 100.0
                        ));
                    }
                }
            }
        }
        for msg in console_msgs {
            self.add_console_message(msg);
        }

        // Update task cache for UI
        self.update_task_cache().await;
    }

    /// Get active plan content for agent context injection
    pub fn get_active_plan_context(&self) -> Option<String> {
        self.active_plan.as_ref().map(|plan| {
            format!(
                "## Active Execution Plan\n\n\
                 **Task:** {}\n\n\
                 **Plan:**\n{}\n\n\
                 Follow this plan step by step. When you complete a step, summarize what was done.",
                plan.task_description, plan.plan_content
            )
        })
    }

    /// Get active plan context with task progress (async version)
    pub async fn get_active_plan_context_with_progress(&self) -> Option<String> {
        let plan = self.active_plan.as_ref()?;

        let (stats, task_tree) = {
            let task_mgr = self.task_manager.read().await;
            (task_mgr.get_stats().await, task_mgr.format_tree().await)
        };

        Some(format!(
            "## Active Execution Plan\n\n\
             **Task:** {}\n\
             **Progress:** {}/{} tasks completed\n\n\
             **Current Tasks:**\n{}\n\n\
             **Plan:**\n{}\n\n\
             Continue working through the plan. Mark tasks complete as you finish them.",
            plan.task_description, stats.completed, stats.total, task_tree, plan.plan_content
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_config_model() {
        // Test that update_config_model updates and saves the config
        let test_model = "claude-haiku-4-5-20251001";

        // This will create/update the config file
        let result = App::update_config_model(test_model);
        assert!(
            result.is_ok(),
            "Failed to update config model: {:?}",
            result.err()
        );

        // Verify the model was persisted by reading it back
        use crate::config::ConfigManager;
        let config_manager = ConfigManager::new().expect("Failed to create config manager");
        let config = config_manager.get();
        assert_eq!(
            config.model, test_model,
            "Model was not persisted correctly"
        );
    }
}
