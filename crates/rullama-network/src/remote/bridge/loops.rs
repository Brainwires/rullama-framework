//! Main communication loops — Realtime WebSocket and HTTP polling.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use super::core::RemoteBridge;
use super::types::RealtimeCredentials;
use crate::remote::protocol::{BackendCommand, StreamChunkType};
use crate::remote::realtime::{RealtimeClient, RealtimeConfig};

impl RemoteBridge {
    /// Main Realtime WebSocket loop (preferred mode)
    pub(super) async fn run_realtime_loop(
        &mut self,
        shutdown_rx: tokio::sync::broadcast::Receiver<()>,
        creds: RealtimeCredentials,
    ) -> Result<()> {
        let user_id = self.user_id.read().await.clone().unwrap_or_default();
        let session_token = self.session_token.read().await.clone().unwrap_or_default();

        let config = RealtimeConfig {
            realtime_url: creds.realtime_url,
            realtime_token: creds.realtime_token,
            channel_name: creds.channel_name,
            user_id: user_id.clone(),
            session_token,
            supabase_anon_key: creds.supabase_anon_key,
            heartbeat_interval_secs: self.config.heartbeat_interval_secs as u64,
            sessions_dir: self.config.sessions_dir.clone(),
            version: self.config.version.clone(),
        };

        let mut client = RealtimeClient::new(config);

        // Create heartbeat channel
        let (heartbeat_tx, heartbeat_rx) =
            tokio::sync::mpsc::channel::<crate::remote::heartbeat::HeartbeatData>(10);

        // Create stream channel for agent output
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::channel::<(String, StreamChunkType, String)>(100);

        // Create sync trigger channel
        let (sync_trigger_tx, mut sync_trigger_rx) = tokio::sync::mpsc::channel::<()>(10);

        // Store channels for command handlers
        *self.stream_tx.write().await = Some(stream_tx);
        *self.sync_trigger_tx.write().await = Some(sync_trigger_tx);

        // Create command channel
        let (command_tx, mut command_rx) = tokio::sync::mpsc::channel::<BackendCommand>(100);

        // Spawn command processor task
        let self_clone = self.clone();
        let command_handle = tokio::spawn(async move {
            tracing::info!("Command processor task started");
            while let Some(cmd) = command_rx.recv().await {
                tracing::info!("Processing Realtime command: {:?}", cmd);
                if let Err(e) = self_clone.handle_backend_command(cmd).await {
                    tracing::error!("Error handling backend command: {}", e);
                }
            }
            tracing::info!("Command processor task ended");
        });

        // Spawn heartbeat collector task
        let heartbeat_collector = Arc::clone(&self.heartbeat_collector);
        let heartbeat_interval = Duration::from_secs(self.config.heartbeat_interval_secs as u64);

        let heartbeat_handle = tokio::spawn(async move {
            tracing::info!(
                "Heartbeat collector task started, interval: {}s",
                heartbeat_interval.as_secs()
            );

            // Send initial heartbeat immediately
            if let Ok(data) = heartbeat_collector.write().await.collect().await {
                tracing::info!(
                    "Sending initial heartbeat with {} agents to frontend",
                    data.agents.len()
                );
                let _ = heartbeat_tx.send(data).await;
            }

            let mut interval = tokio::time::interval(heartbeat_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Ok(data) = heartbeat_collector.write().await.collect().await {
                            tracing::info!("Sending heartbeat with {} agents to frontend", data.agents.len());
                            let _ = heartbeat_tx.send(data).await;
                        }
                    }
                    Some(()) = sync_trigger_rx.recv() => {
                        tracing::info!("Sync trigger received, sending immediate heartbeat");
                        if let Ok(data) = heartbeat_collector.write().await.collect().await {
                            tracing::info!("Sending sync heartbeat with {} agents to frontend", data.agents.len());
                            let _ = heartbeat_tx.send(data).await;
                        }
                    }
                }
            }
        });

        // Connect and run
        client
            .connect(shutdown_rx, heartbeat_rx, stream_rx, command_tx)
            .await?;

        // Clean up
        *self.stream_tx.write().await = None;
        *self.sync_trigger_tx.write().await = None;
        heartbeat_handle.abort();
        command_handle.abort();

        Ok(())
    }

    /// Main polling loop
    pub(super) async fn run_polling_loop(
        &mut self,
        mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
    ) -> Result<()> {
        let heartbeat_interval = Duration::from_secs(self.config.heartbeat_interval_secs as u64);
        let mut heartbeat_timer = tokio::time::interval(heartbeat_interval);

        // Initial heartbeat
        self.send_heartbeat_and_process_commands().await?;

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("Received shutdown signal");
                    break;
                }
                _ = heartbeat_timer.tick() => {
                    if let Err(e) = self.send_heartbeat_and_process_commands().await {
                        tracing::error!("Heartbeat failed: {}", e);
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Send heartbeat and process any commands returned
    pub(super) async fn send_heartbeat_and_process_commands(&self) -> Result<()> {
        let session_token = self.session_token.read().await.clone().unwrap_or_default();

        let heartbeat_data = self.heartbeat_collector.write().await.collect().await?;

        // Drain queued command results
        let command_results: Vec<crate::remote::protocol::RemoteMessage> = {
            let mut queue = self.command_result_queue.write().await;
            let msgs = std::mem::take(&mut *queue);
            if !msgs.is_empty() {
                tracing::info!("Sending {} command results in heartbeat", msgs.len());
            }
            msgs
        };

        let heartbeat_body = serde_json::json!({
            "session_token": session_token,
            "agents": heartbeat_data.agents,
            "system_load": heartbeat_data.system_load,
            "messages": command_results,
            "hostname": gethostname::gethostname().to_string_lossy().to_string(),
            "os": std::env::consts::OS.to_string(),
            "version": self.config.version.clone(),
        });

        let url = format!("{}/api/remote/heartbeat", self.config.backend_url);

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&heartbeat_body)
            .send()
            .await
            .context("Heartbeat request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Heartbeat failed: {} - {}", status, body);
        }

        #[cfg(feature = "telemetry")]
        if let Some(ref collector) = self.analytics_collector {
            use rullama_telemetry::AnalyticsEvent;
            let body_bytes = serde_json::to_vec(&heartbeat_body).unwrap_or_default();
            collector.record(AnalyticsEvent::NetworkMessage {
                session_id: None,
                protocol: "remote-bridge".to_string(),
                direction: "outbound".to_string(),
                bytes: body_bytes.len() as u64,
                success: true,
                timestamp: chrono::Utc::now(),
            });
        }

        let response_body: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse heartbeat response")?;

        // Process any commands from the response
        if let Some(commands) = response_body.get("commands").and_then(|v| v.as_array()) {
            if !commands.is_empty() {
                tracing::info!(
                    "Received {} commands from backend: {:?}",
                    commands.len(),
                    commands
                );
            }
            for cmd_value in commands {
                match serde_json::from_value::<BackendCommand>(cmd_value.clone()) {
                    Ok(cmd) => {
                        tracing::info!("Processing backend command: {:?}", cmd);
                        if let Err(e) = self.handle_backend_command(cmd).await {
                            tracing::error!("Error handling backend command: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse backend command {:?}: {}", cmd_value, e);
                    }
                }
            }
        }

        Ok(())
    }
}
