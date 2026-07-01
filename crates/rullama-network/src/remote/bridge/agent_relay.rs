//! Agent IPC relay — subscribe/unsubscribe, input relay, spawn, and message conversion.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use super::core::RemoteBridge;
use super::types::AgentSubscription;
use crate::ipc::{AgentMessage, Handshake, HandshakeResponse, IpcConnection, ViewerMessage};
use crate::remote::protocol::StreamChunkType;

impl RemoteBridge {
    /// Start an agent reader task to stream output back to backend
    pub(super) async fn start_agent_reader(&self, agent_id: &str) {
        tracing::info!("start_agent_reader called for agent: {}", agent_id);

        // Acquire write lock for the check to avoid a race where two concurrent
        // calls both see no entry and both proceed to start a reader.
        {
            let tasks = self.subscription_tasks.write().await;
            if tasks.contains_key(agent_id) {
                tracing::debug!("Agent reader already running for {}", agent_id);
                return;
            }
            // Drop the write lock before the async IPC work below.
        }

        let sessions_dir = &self.config.sessions_dir;

        // Connect using bridge-internal IPC with injected sessions_dir.
        // Apply a timeout so a wedged agent socket doesn't block the bridge indefinitely.
        tracing::info!("Connecting to agent {} via IPC...", agent_id);
        let mut conn = match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            IpcConnection::connect_to_agent(sessions_dir, agent_id),
        )
        .await
        {
            Ok(Ok(c)) => {
                tracing::info!("Successfully connected to agent {}", agent_id);
                c
            }
            Ok(Err(e)) => {
                tracing::error!(
                    "Failed to connect to agent {} for streaming: {}",
                    agent_id,
                    e
                );
                return;
            }
            Err(_) => {
                tracing::warn!(
                    "Timed out connecting to agent {} IPC socket after 5s, skipping",
                    agent_id
                );
                return;
            }
        };

        // Read session token using bridge-internal parameterized function
        let session_token = match crate::ipc::socket::read_session_token(sessions_dir, agent_id) {
            Ok(Some(token)) => token,
            Ok(None) => {
                tracing::error!("No session token found for agent {}", agent_id);
                return;
            }
            Err(e) => {
                tracing::error!("Failed to read session token for agent {}: {}", agent_id, e);
                return;
            }
        };

        // Perform handshake
        tracing::info!("Sending handshake with token to agent {}", agent_id);
        let handshake = Handshake::reattach(agent_id.to_string(), session_token);
        if let Err(e) = conn.writer.write(&handshake).await {
            tracing::error!("Failed to send handshake to agent {}: {}", agent_id, e);
            return;
        }

        // Wait for handshake response
        tracing::info!("Waiting for handshake response from agent {}", agent_id);
        let response: HandshakeResponse = match conn.reader.read().await {
            Ok(Some(r)) => r,
            Ok(None) => {
                tracing::error!("Agent {} closed connection during handshake", agent_id);
                return;
            }
            Err(e) => {
                tracing::error!(
                    "Failed to read handshake response from agent {}: {}",
                    agent_id,
                    e
                );
                return;
            }
        };

        if !response.accepted {
            tracing::error!(
                "Handshake rejected by agent {}: {:?}",
                agent_id,
                response.error
            );
            return;
        }
        tracing::info!("Handshake accepted by agent {}", agent_id);

        // Request conversation sync
        tracing::info!("Sending SyncRequest to agent {}", agent_id);
        if let Err(e) = conn.writer.write(&ViewerMessage::SyncRequest).await {
            tracing::error!("Failed to send SyncRequest to agent {}: {}", agent_id, e);
        } else {
            tracing::info!("SyncRequest sent successfully to agent {}", agent_id);
        }

        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        let agent_id_owned = agent_id.to_string();
        let subscriptions = Arc::clone(&self.subscriptions);
        let stream_tx = Arc::clone(&self.stream_tx);

        // Create channel for sending messages to this agent
        let (writer_tx, mut writer_rx) = tokio::sync::mpsc::channel::<ViewerMessage>(32);

        // Spawn reader/writer task
        tracing::info!("Spawning reader task for agent {}", agent_id);
        let task_handle = tokio::spawn(async move {
            tracing::info!("Agent reader task started for {}", agent_id_owned);
            let (mut reader, mut writer) = (conn.reader, conn.writer);
            let mut cancel_rx = cancel_rx;

            loop {
                tokio::select! {
                    _ = &mut cancel_rx => {
                        tracing::debug!("Agent reader for {} cancelled", agent_id_owned);
                        break;
                    }
                    Some(msg) = writer_rx.recv() => {
                        tracing::info!("Sending ViewerMessage to agent {}: {:?}", agent_id_owned, std::mem::discriminant(&msg));
                        if let Err(e) = writer.write(&msg).await {
                            tracing::error!("Failed to send message to agent {}: {}", agent_id_owned, e);
                            break;
                        }
                    }
                    result = reader.read::<AgentMessage>() => {
                        match result {
                            Ok(Some(msg)) => {
                                tracing::info!("Received AgentMessage from {}: {:?}", agent_id_owned, std::mem::discriminant(&msg));

                                if !subscriptions.read().await.contains(&agent_id_owned) {
                                    tracing::debug!("Agent {} no longer subscribed, stopping reader", agent_id_owned);
                                    break;
                                }

                                if let Some((chunk_type, content)) = convert_agent_message_to_stream(&msg) {
                                    tracing::info!("Sending stream message for {}: type={:?}, content_len={}",
                                        agent_id_owned, chunk_type, content.len());

                                    if let Some(tx) = stream_tx.read().await.as_ref() {
                                        if let Err(e) = tx.send((agent_id_owned.clone(), chunk_type, content)).await {
                                            tracing::error!("Failed to send stream via Realtime: {}", e);
                                        } else {
                                            tracing::debug!("Stream message sent via Realtime");
                                        }
                                    } else {
                                        tracing::warn!("Realtime stream channel not available, dropping message");
                                    }
                                } else {
                                    tracing::debug!("AgentMessage not converted to stream chunk (filtered out)");
                                }
                            }
                            Ok(None) => {
                                tracing::info!("Agent {} disconnected", agent_id_owned);
                                break;
                            }
                            Err(e) => {
                                tracing::error!("Error reading from agent {}: {}", agent_id_owned, e);
                                break;
                            }
                        }
                    }
                }
            }
            tracing::info!("Agent reader task ended for {}", agent_id_owned);
        });

        // Store subscription with writer channel
        self.subscription_tasks.write().await.insert(
            agent_id.to_string(),
            AgentSubscription {
                cancel_tx,
                task_handle,
                writer_tx,
            },
        );

        tracing::info!("Started agent reader for {}", agent_id);
    }

    /// Stop an agent reader task
    pub(super) async fn stop_agent_reader(&self, agent_id: &str) {
        if let Some(sub) = self.subscription_tasks.write().await.remove(agent_id) {
            let _ = sub.cancel_tx.send(());
            sub.task_handle.abort();
            tracing::info!("Stopped agent reader for {}", agent_id);
        }
    }

    /// Request history sync from an agent
    pub(super) async fn request_history_sync(&self, agent_id: &str) {
        tracing::info!("Requesting history sync for agent: {}", agent_id);

        let writer_tx = {
            let tasks = self.subscription_tasks.read().await;
            tasks.get(agent_id).map(|sub| sub.writer_tx.clone())
        };

        let writer_tx = match writer_tx {
            Some(tx) => tx,
            None => {
                tracing::warn!(
                    "No active subscription for agent {}, cannot request history sync",
                    agent_id
                );
                return;
            }
        };

        if let Err(e) = writer_tx.send(ViewerMessage::SyncRequest).await {
            tracing::error!("Failed to send SyncRequest to agent {}: {}", agent_id, e);
            return;
        }

        tracing::info!(
            "SyncRequest sent to agent {} via persistent connection",
            agent_id
        );
    }

    /// Relay user input to an agent
    pub(super) async fn relay_input_to_agent(
        &self,
        agent_id: &str,
        content: &str,
    ) -> Result<serde_json::Value> {
        let writer_tx = {
            let tasks = self.subscription_tasks.read().await;
            tasks.get(agent_id).map(|sub| sub.writer_tx.clone())
        };

        let writer_tx = writer_tx
            .ok_or_else(|| anyhow::anyhow!("No active subscription for agent {}", agent_id))?;

        let msg = ViewerMessage::UserInput {
            content: content.to_string(),
            context_files: vec![],
        };

        writer_tx
            .send(msg)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send input to agent: {}", e))?;

        Ok(serde_json::json!({
            "agent_id": agent_id,
            "input_sent": true,
        }))
    }

    /// Relay slash command to an agent
    pub(super) async fn relay_slash_command_to_agent(
        &self,
        agent_id: &str,
        command: &str,
        args: &[String],
    ) -> Result<serde_json::Value> {
        let writer_tx = {
            let tasks = self.subscription_tasks.read().await;
            tasks.get(agent_id).map(|sub| sub.writer_tx.clone())
        };

        let writer_tx = writer_tx
            .ok_or_else(|| anyhow::anyhow!("No active subscription for agent {}", agent_id))?;

        let msg = ViewerMessage::SlashCommand {
            command: command.to_string(),
            args: args.to_vec(),
        };

        writer_tx
            .send(msg)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send command to agent: {}", e))?;

        Ok(serde_json::json!({
            "agent_id": agent_id,
            "command": command,
            "command_sent": true,
        }))
    }

    /// Relay cancel to an agent
    pub(super) async fn relay_cancel_to_agent(&self, agent_id: &str) -> Result<serde_json::Value> {
        let writer_tx = {
            let tasks = self.subscription_tasks.read().await;
            tasks.get(agent_id).map(|sub| sub.writer_tx.clone())
        };

        let writer_tx = writer_tx
            .ok_or_else(|| anyhow::anyhow!("No active subscription for agent {}", agent_id))?;

        let msg = ViewerMessage::Cancel;
        writer_tx
            .send(msg)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send cancel to agent: {}", e))?;

        Ok(serde_json::json!({
            "agent_id": agent_id,
            "cancel_sent": true,
        }))
    }

    /// Spawn a new agent (session)
    ///
    /// Delegates to the injected `AgentSpawner` trait for actual process creation.
    /// The bridge handles session ID generation and basic path validation.
    pub(super) async fn spawn_new_agent(
        &self,
        model: Option<String>,
        working_directory: Option<String>,
    ) -> Result<serde_json::Value> {
        let agent_spawner = self
            .agent_spawner
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No agent spawner configured"))?;

        // Validate and canonicalize working_directory
        let validated_working_dir = if let Some(ref dir) = working_directory {
            let path = PathBuf::from(dir);

            if !path.exists() {
                anyhow::bail!("Working directory does not exist: {}", dir);
            }
            if !path.is_dir() {
                anyhow::bail!("Working directory is not a directory: {}", dir);
            }

            let canonical = path
                .canonicalize()
                .context(format!("Failed to canonicalize working directory: {}", dir))?;

            Some(canonical)
        } else {
            None
        };

        // Generate a cryptographically secure session ID
        use rand::Rng;
        let mut random_bytes = [0u8; 16];
        rand::rng().fill_bytes(&mut random_bytes);
        let session_id = format!("session-{}", hex::encode(random_bytes));

        // Delegate to spawner
        tracing::info!("Spawning new session via remote: {}", session_id);
        let socket_path = agent_spawner
            .spawn_agent(&session_id, model, validated_working_dir)
            .await?;

        Ok(serde_json::json!({
            "session_id": session_id,
            "socket_path": socket_path.to_string_lossy(),
            "status": "spawned",
        }))
    }
}

/// Convert an AgentMessage to a stream chunk (chunk_type, content)
pub(super) fn convert_agent_message_to_stream(
    msg: &AgentMessage,
) -> Option<(StreamChunkType, String)> {
    match msg {
        AgentMessage::StreamChunk { text } => Some((StreamChunkType::Text, text.clone())),
        AgentMessage::StreamEnd { .. } => Some((StreamChunkType::Complete, String::new())),
        AgentMessage::ToolCallStart { name, input, .. } => {
            let content = format!(
                "Tool: {} - {}",
                name,
                serde_json::to_string(input).unwrap_or_default()
            );
            Some((StreamChunkType::ToolCall, content))
        }
        AgentMessage::ToolProgress { name, message, .. } => {
            Some((StreamChunkType::Text, format!("[{}] {}", name, message)))
        }
        AgentMessage::ToolResult {
            name,
            output,
            error,
            ..
        } => {
            let content = if let Some(err) = error {
                format!("{}: Error: {}", name, err)
            } else if let Some(out) = output {
                format!("{}: {}", name, out)
            } else {
                format!("{}: (no output)", name)
            };
            Some((StreamChunkType::ToolResult, content))
        }
        AgentMessage::Error { message, .. } => Some((StreamChunkType::Error, message.clone())),
        AgentMessage::StatusUpdate { status } => Some((StreamChunkType::System, status.clone())),
        AgentMessage::MessageAdded { message } => {
            if message.role == "user" {
                Some((StreamChunkType::UserInput, message.content.clone()))
            } else {
                None
            }
        }
        AgentMessage::ConversationSync { messages, .. } => {
            let history_json = serde_json::to_string(messages).unwrap_or_else(|_| "[]".to_string());
            Some((StreamChunkType::History, history_json))
        }
        AgentMessage::SlashCommandResult {
            command,
            success,
            output,
            action_taken,
            error,
            blocked,
            ..
        } => {
            let content = if *blocked {
                format!(
                    "Command /{} blocked: {}",
                    command,
                    error.as_deref().unwrap_or("security policy")
                )
            } else if *success {
                if let Some(out) = output {
                    format!("/{}: {}", command, out)
                } else if let Some(action) = action_taken {
                    format!("/{}: {}", command, action)
                } else {
                    format!("/{}: done", command)
                }
            } else {
                format!(
                    "/{} failed: {}",
                    command,
                    error.as_deref().unwrap_or("unknown error")
                )
            };
            Some((StreamChunkType::System, content))
        }
        // Not exposed to remote bridge
        AgentMessage::TaskUpdate { .. }
        | AgentMessage::Toast { .. }
        | AgentMessage::SealStatus { .. }
        | AgentMessage::LockResult { .. }
        | AgentMessage::LockReleased { .. }
        | AgentMessage::LockStatus { .. }
        | AgentMessage::LockChanged { .. }
        | AgentMessage::Ack { .. }
        | AgentMessage::Exiting { .. }
        | AgentMessage::AgentSpawned { .. }
        | AgentMessage::AgentList { .. }
        | AgentMessage::AgentExiting { .. }
        | AgentMessage::ParentSignalReceived { .. }
        | AgentMessage::PlanModeEntered { .. }
        | AgentMessage::PlanModeExited { .. }
        | AgentMessage::PlanModeSync { .. }
        | AgentMessage::PlanModeMessageAdded { .. }
        | AgentMessage::PlanModeStreamChunk { .. }
        | AgentMessage::PlanModeStreamEnd { .. } => None,
    }
}
