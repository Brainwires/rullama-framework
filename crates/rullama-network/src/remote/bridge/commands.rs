//! Backend command dispatch and result queuing.

use anyhow::Result;

use super::core::RemoteBridge;
use crate::remote::protocol::{BackendCommand, RemoteMessage};

impl RemoteBridge {
    /// Handle a command from the backend
    pub(super) async fn handle_backend_command(&self, cmd: BackendCommand) -> Result<()> {
        match cmd {
            BackendCommand::Ping { timestamp } => {
                self.queue_command_result_msg(RemoteMessage::Pong { timestamp })
                    .await?;
            }

            BackendCommand::RequestSync => {
                tracing::info!("Backend requested sync, triggering immediate heartbeat");
                if let Some(tx) = self.sync_trigger_tx.read().await.as_ref()
                    && let Err(e) = tx.send(()).await
                {
                    tracing::error!("Failed to trigger sync: {}", e);
                }
            }

            BackendCommand::Subscribe { agent_id } => {
                tracing::info!("Web client subscribed to agent: {}", agent_id);
                self.subscriptions.write().await.insert(agent_id.clone());
                self.start_agent_reader(&agent_id).await;
                self.request_history_sync(&agent_id).await;
            }

            BackendCommand::Unsubscribe { agent_id } => {
                tracing::info!("Web client unsubscribed from agent: {}", agent_id);
                self.subscriptions.write().await.remove(&agent_id);
                self.stop_agent_reader(&agent_id).await;
            }

            BackendCommand::SendInput {
                command_id,
                agent_id,
                content,
            } => {
                let result = self.relay_input_to_agent(&agent_id, &content).await;
                self.queue_command_result(&command_id, result).await?;
            }

            BackendCommand::SlashCommand {
                command_id,
                agent_id,
                command,
                args,
            } => {
                let result = self
                    .relay_slash_command_to_agent(&agent_id, &command, &args)
                    .await;
                self.queue_command_result(&command_id, result).await?;
            }

            BackendCommand::CancelOperation {
                command_id,
                agent_id,
            } => {
                let result = self.relay_cancel_to_agent(&agent_id).await;
                self.queue_command_result(&command_id, result).await?;
            }

            BackendCommand::SpawnAgent {
                command_id,
                model,
                working_directory,
            } => {
                let result = self.spawn_new_agent(model, working_directory).await;
                self.queue_command_result(&command_id, result).await?;
            }

            BackendCommand::Disconnect { reason } => {
                tracing::info!("Backend requested disconnect: {}", reason);
            }

            // Attachment Commands
            BackendCommand::AttachmentUpload {
                command_id,
                agent_id,
                attachment_id,
                filename,
                mime_type,
                size,
                compressed,
                compression_algorithm,
                chunks_total,
            } => {
                tracing::info!(
                    "Starting attachment upload: {} ({} bytes, {} chunks)",
                    filename,
                    size,
                    chunks_total
                );

                let result = self
                    .attachment_receiver
                    .start_upload(
                        command_id.clone(),
                        agent_id,
                        attachment_id.clone(),
                        filename,
                        mime_type,
                        size,
                        compressed,
                        compression_algorithm,
                        chunks_total,
                    )
                    .await;

                match result {
                    Ok(()) => {
                        self.queue_command_result(
                            &command_id,
                            Ok(serde_json::json!({
                                "attachment_id": attachment_id,
                                "status": "started"
                            })),
                        )
                        .await?;
                    }
                    Err(e) => {
                        self.queue_command_result(&command_id, Err(e)).await?;
                    }
                }
            }

            BackendCommand::AttachmentChunk {
                attachment_id,
                chunk_index,
                data,
                is_final,
            } => {
                tracing::debug!(
                    "Receiving attachment chunk: {} (index {})",
                    attachment_id,
                    chunk_index
                );

                match self
                    .attachment_receiver
                    .receive_chunk(&attachment_id, chunk_index, &data, is_final)
                    .await
                {
                    Ok(all_received) => {
                        if all_received {
                            tracing::info!("All chunks received for attachment: {}", attachment_id);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to receive chunk for {}: {}", attachment_id, e);
                        self.attachment_receiver.cancel_upload(&attachment_id).await;
                    }
                }
            }

            BackendCommand::AttachmentComplete {
                attachment_id,
                checksum,
            } => {
                tracing::info!(
                    "Completing attachment upload: {} (checksum: {})",
                    attachment_id,
                    checksum
                );

                match self
                    .attachment_receiver
                    .complete_upload(&attachment_id, &checksum)
                    .await
                {
                    Ok(file_path) => {
                        let path_str = file_path.display().to_string();
                        tracing::info!("Attachment saved to: {}", path_str);

                        self.queue_command_result_msg(RemoteMessage::AttachmentReceived {
                            attachment_id: attachment_id.clone(),
                            success: true,
                            file_path: Some(path_str),
                            error: None,
                        })
                        .await?;
                    }
                    Err(e) => {
                        tracing::error!("Failed to complete attachment: {}", e);

                        self.queue_command_result_msg(RemoteMessage::AttachmentReceived {
                            attachment_id: attachment_id.clone(),
                            success: false,
                            file_path: None,
                            error: Some(e.to_string()),
                        })
                        .await?;
                    }
                }
            }

            BackendCommand::PermissionResponse {
                request_id,
                approved,
                remember_for_session,
                always_allow,
            } => {
                tracing::info!(
                    "Permission response for {}: approved={}",
                    request_id,
                    approved
                );
                // We don't have the tool_name here, but the CLI side tracks that.
                // Use resolve() without tool name; the CLI caller handles session memory.
                self.permission_relay
                    .resolve(
                        &request_id,
                        crate::remote::permission_relay::PermissionDecision {
                            approved,
                            remember_for_session,
                            always_allow,
                        },
                    )
                    .await;
            }

            BackendCommand::Authenticated { .. } | BackendCommand::AuthenticationFailed { .. } => {
                tracing::warn!("Unexpected auth message after authentication");
            }
        }

        Ok(())
    }

    /// Queue a command result to send with the next heartbeat
    pub(super) async fn queue_command_result(
        &self,
        command_id: &str,
        result: Result<serde_json::Value>,
    ) -> Result<()> {
        let msg = match result {
            Ok(value) => RemoteMessage::CommandResult {
                command_id: command_id.to_string(),
                success: true,
                result: Some(value),
                error: None,
            },
            Err(e) => RemoteMessage::CommandResult {
                command_id: command_id.to_string(),
                success: false,
                result: None,
                error: Some(e.to_string()),
            },
        };

        self.queue_command_result_msg(msg).await
    }
}
