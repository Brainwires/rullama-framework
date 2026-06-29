//! Supabase Realtime WebSocket Client
//!
//! Provides bidirectional communication with the rullama-studio backend
//! via Supabase Realtime WebSocket channels.
//!
//! # Architecture
//!
//! - CLI connects to Supabase Realtime WebSocket endpoint
//! - Subscribes to channel `cli:{userId}` for bidirectional communication
//! - Server and CLI both send messages via broadcast events
//! - Eliminates HTTP polling latency for commands
//!
//! All CLI-specific dependencies have been removed:
//! - `crate::ipc::list_agent_sessions_with_metadata()` → uses `sessions_dir` from config
//! - `env!("CARGO_PKG_VERSION")` → uses `version` from config

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use url::Url;

use super::protocol::{BackendCommand, RemoteAgentInfo, StreamChunkType};

/// Phoenix protocol heartbeat interval in seconds (must be < 60s to keep connection alive).
const PHOENIX_HEARTBEAT_INTERVAL_SECS: u64 = 25;

/// Supabase Realtime connection configuration
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// WebSocket URL (wss://...)
    pub realtime_url: String,
    /// Supabase JWT for authentication
    pub realtime_token: String,
    /// Channel name to subscribe to (cli:{userId})
    pub channel_name: String,
    /// User ID (for message routing)
    pub user_id: String,
    /// Session token (for backend tracking)
    pub session_token: String,
    /// Supabase anon key (used as apikey param in WebSocket URL for Kong auth)
    pub supabase_anon_key: String,
    /// Heartbeat interval in seconds (for sending agent status to frontend)
    pub heartbeat_interval_secs: u64,
    /// Sessions directory for agent discovery (injected, replaces PlatformPaths)
    pub sessions_dir: PathBuf,
    /// CLI version string (injected, replaces env!("CARGO_PKG_VERSION"))
    pub version: String,
}

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealtimeState {
    /// Not connected to the Realtime server.
    Disconnected,
    /// Connection in progress.
    Connecting,
    /// WebSocket connected but channel not yet joined.
    Connected,
    /// Channel joined and ready for messages.
    Subscribed,
    /// Gracefully shutting down.
    ShuttingDown,
}

/// Message types for Supabase Realtime protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum PhoenixMessage {
    /// Join a channel
    #[serde(rename = "phx_join")]
    PhxJoin {
        /// Channel topic to join.
        topic: String,
        /// Join parameters.
        payload: serde_json::Value,
        /// Message reference ID.
        #[serde(rename = "ref")]
        msg_ref: String,
    },
    /// Reply to a message
    #[serde(rename = "phx_reply")]
    PhxReply {
        /// Channel topic.
        topic: String,
        /// Reply payload with status.
        payload: PhxReplyPayload,
        /// Message reference ID.
        #[serde(rename = "ref")]
        msg_ref: String,
    },
    /// Heartbeat (keep-alive)
    #[serde(rename = "heartbeat")]
    Heartbeat {
        /// Channel topic (usually "phoenix").
        topic: String,
        /// Heartbeat payload.
        payload: serde_json::Value,
        /// Message reference ID.
        #[serde(rename = "ref")]
        msg_ref: String,
    },
    /// Broadcast message
    #[serde(rename = "broadcast")]
    Broadcast {
        /// Channel topic.
        topic: String,
        /// Broadcast payload containing the event data.
        payload: BroadcastPayload,
        /// Optional message reference ID.
        #[serde(rename = "ref")]
        msg_ref: Option<String>,
    },
    /// Presence state
    #[serde(rename = "presence_state")]
    PresenceState {
        /// Channel topic.
        topic: String,
        /// Presence state data.
        payload: serde_json::Value,
        /// Optional message reference ID.
        #[serde(rename = "ref")]
        msg_ref: Option<String>,
    },
    /// Presence diff
    #[serde(rename = "presence_diff")]
    PresenceDiff {
        /// Channel topic.
        topic: String,
        /// Presence diff data.
        payload: serde_json::Value,
        /// Optional message reference ID.
        #[serde(rename = "ref")]
        msg_ref: Option<String>,
    },
}

/// Payload for Phoenix reply messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhxReplyPayload {
    /// Reply status (e.g., "ok" or "error").
    pub status: String,
    /// Response data.
    #[serde(default)]
    pub response: serde_json::Value,
}

/// Payload for broadcast messages on a Realtime channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadcastPayload {
    /// Broadcast type identifier.
    #[serde(rename = "type")]
    pub broadcast_type: String,
    /// Event name.
    pub event: String,
    /// Event payload data.
    pub payload: serde_json::Value,
}

/// Remote Realtime message types (matching TypeScript protocol)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RemoteRealtimeMessage {
    /// CLI to Backend: Initial registration
    #[serde(rename = "remote.register")]
    Register {
        /// Registration payload.
        payload: RegisterPayload,
    },
    /// CLI to Backend: Status update with agents
    #[serde(rename = "remote.heartbeat")]
    Heartbeat {
        /// Heartbeat payload with agent info.
        payload: HeartbeatPayload,
    },
    /// CLI to Backend: Agent output stream
    #[serde(rename = "remote.stream")]
    Stream {
        /// Stream chunk payload.
        payload: StreamPayload,
    },
    /// CLI to Backend: Result of a command
    #[serde(rename = "remote.command_result")]
    CommandResult {
        /// Command result payload.
        payload: CommandResultPayload,
    },
    /// CLI to Backend: Agent event
    #[serde(rename = "remote.event")]
    Event {
        /// Agent event payload.
        payload: EventPayload,
    },
    /// Backend to CLI: Command to execute
    #[serde(rename = "remote.command")]
    Command {
        /// Command payload from backend.
        payload: CommandPayload,
    },
    /// Backend to CLI: Ping
    #[serde(rename = "remote.ping")]
    Ping {
        /// Ping payload with timestamp.
        payload: PingPongPayload,
    },
    /// CLI to Backend: Pong
    #[serde(rename = "remote.pong")]
    Pong {
        /// Pong payload with timestamp.
        payload: PingPongPayload,
    },
    /// CLI to Backend: Graceful disconnect notification
    #[serde(rename = "remote.disconnect")]
    Disconnect {
        /// Disconnect payload with reason.
        payload: DisconnectPayload,
    },
}

/// Payload for CLI registration with the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisterPayload {
    /// Client hostname.
    pub hostname: String,
    /// Client operating system.
    pub os: String,
    /// CLI version string.
    pub version: String,
    /// Session token for authentication.
    pub session_token: String,
    /// Include agents in register so frontend gets them immediately
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agents: Vec<RemoteAgentInfo>,
    /// System load
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_load: Option<f32>,
}

/// Payload for periodic heartbeat messages with agent status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeartbeatPayload {
    /// List of active agents.
    pub agents: Vec<RemoteAgentInfo>,
    /// Current system load (0.0-1.0).
    pub system_load: f32,
    /// Client hostname.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    /// Client operating system.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    /// CLI version string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Payload for agent output stream chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamPayload {
    /// ID of the agent producing the stream.
    pub agent_id: String,
    /// Type of stream chunk.
    pub chunk_type: StreamChunkType,
    /// Chunk content text.
    pub content: String,
}

/// Payload for command execution results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandResultPayload {
    /// ID of the command being responded to.
    pub command_id: String,
    /// Whether the command succeeded.
    pub success: bool,
    /// Result data if successful.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Payload for agent event notifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventPayload {
    /// Type of agent event.
    pub event_type: String,
    /// ID of the agent this event relates to.
    pub agent_id: String,
    /// Event-specific data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Payload for commands received from the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandPayload {
    /// Unique command identifier.
    pub command_id: String,
    /// Type of command (e.g., "send_input", "slash_command").
    pub command_type: String,
    /// Target agent ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Input content for send_input commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Slash command name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Slash command arguments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    /// Model for spawn_agent commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Working directory for spawn_agent commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_directory: Option<String>,
    /// Reason for certain commands (e.g., disconnect).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Payload for ping/pong messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PingPongPayload {
    /// Server timestamp for round-trip measurement.
    pub server_timestamp: i64,
}

/// Payload for graceful disconnect notifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisconnectPayload {
    /// Reason for disconnection.
    pub reason: String,
    /// Hostname of disconnecting bridge (for multi-bridge support)
    pub hostname: String,
}

/// Supabase Realtime WebSocket client
pub struct RealtimeClient {
    config: RealtimeConfig,
    state: Arc<RwLock<RealtimeState>>,
    /// Channel for outgoing messages
    outgoing_tx: Option<mpsc::Sender<RemoteRealtimeMessage>>,
    /// Message reference counter
    msg_ref: Arc<RwLock<u64>>,
}

impl RealtimeClient {
    /// Create a new Realtime client
    pub fn new(config: RealtimeConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(RealtimeState::Disconnected)),
            outgoing_tx: None,
            msg_ref: Arc::new(RwLock::new(0)),
        }
    }

    /// Get the next message reference
    async fn next_ref(&self) -> String {
        let mut ref_num = self.msg_ref.write().await;
        *ref_num += 1;
        ref_num.to_string()
    }

    /// Get current connection state
    pub async fn state(&self) -> RealtimeState {
        *self.state.read().await
    }

    /// Check if connected and subscribed
    pub async fn is_ready(&self) -> bool {
        *self.state.read().await == RealtimeState::Subscribed
    }

    /// Connect to Supabase Realtime and run the message loop
    ///
    /// - `shutdown_rx`: Signal to gracefully shut down
    /// - `heartbeat_rx`: Receives heartbeat data (with hostname, os, version) to broadcast to frontend
    /// - `stream_rx`: Receives stream messages (agent_id, chunk_type, content) to broadcast
    /// - `command_tx`: Channel to send commands received from frontend for processing
    pub async fn connect(
        &mut self,
        mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
        mut heartbeat_rx: mpsc::Receiver<super::heartbeat::HeartbeatData>,
        mut stream_rx: mpsc::Receiver<(String, StreamChunkType, String)>,
        command_tx: mpsc::Sender<BackendCommand>,
    ) -> Result<()> {
        *self.state.write().await = RealtimeState::Connecting;

        // Build WebSocket URL with anon key (required by Kong) and protocol version
        let mut url = Url::parse(&self.config.realtime_url)?;
        url.query_pairs_mut()
            .append_pair("apikey", &self.config.supabase_anon_key)
            .append_pair("vsn", "1.0.0");

        tracing::info!(
            "Connecting to Supabase Realtime: {}",
            url.host_str().unwrap_or("unknown")
        );

        // Build a WebSocket request with auth header
        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(url.as_str())
            .header(
                "Authorization",
                format!("Bearer {}", self.config.realtime_token),
            )
            .body(())
            .context("Failed to build WebSocket request")?;

        // Connect WebSocket
        let (ws_stream, _response): (
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            _,
        ) = match connect_async(request).await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("WebSocket connection error: {:?}", e);
                return Err(anyhow::anyhow!(
                    "Failed to connect to Supabase Realtime: {}",
                    e
                ));
            }
        };

        *self.state.write().await = RealtimeState::Connected;
        tracing::info!("Connected to Supabase Realtime");

        let (mut write, mut read) = ws_stream.split();

        // Create channel for outgoing messages
        let (outgoing_tx, mut outgoing_rx) = mpsc::channel::<RemoteRealtimeMessage>(100);

        self.outgoing_tx = Some(outgoing_tx);

        // Join the channel
        let join_ref = self.next_ref().await;
        let channel_topic = format!("realtime:{}", self.config.channel_name);
        let join_msg = serde_json::json!({
            "topic": channel_topic,
            "event": "phx_join",
            "payload": {
                "config": {
                    "broadcast": {
                        "self": false
                    },
                    "presence": {
                        "key": ""
                    }
                }
            },
            "ref": join_ref
        });

        write
            .send(Message::Text(serde_json::to_string(&join_msg)?.into()))
            .await
            .context("Failed to send join message")?;

        tracing::info!("Sent join request for channel: {}", channel_topic);

        // Main message loop with Phoenix heartbeat
        let state = Arc::clone(&self.state);
        let channel_topic_clone = channel_topic.clone();
        let user_id = self.config.user_id.clone();
        let session_token = self.config.session_token.clone();
        let sessions_dir = self.config.sessions_dir.clone();
        let version = self.config.version.clone();

        // Phoenix heartbeat interval (must be < 60s to keep connection alive)
        let mut phoenix_heartbeat =
            tokio::time::interval(Duration::from_secs(PHOENIX_HEARTBEAT_INTERVAL_SECS));

        // Track if we've sent the initial register message
        let mut register_sent = false;

        loop {
            // Check if we just became subscribed and need to send register
            if !register_sent && *state.read().await == RealtimeState::Subscribed {
                register_sent = true;
                tracing::info!("Channel subscribed, sending register message to frontend");

                // Get current agents to include in register message
                // Uses bridge-internal discovery with injected sessions_dir
                let agents =
                    crate::ipc::discovery::list_agent_sessions_with_metadata(&sessions_dir)
                        .unwrap_or_default()
                        .into_iter()
                        .map(RemoteAgentInfo::from)
                        .collect::<Vec<_>>();

                let register_msg = RemoteRealtimeMessage::Register {
                    payload: RegisterPayload {
                        hostname: gethostname::gethostname().to_string_lossy().to_string(),
                        os: std::env::consts::OS.to_string(),
                        version: version.clone(),
                        session_token: session_token.clone(),
                        agents,
                        system_load: None,
                    },
                };

                if let Err(e) = self
                    .send_broadcast(&mut write, &channel_topic_clone, &user_id, register_msg)
                    .await
                {
                    tracing::warn!("Failed to send register message: {}", e);
                } else {
                    tracing::info!("Register message sent to frontend");
                }
            }

            tokio::select! {
                // Shutdown signal
                _ = shutdown_rx.recv() => {
                    tracing::info!("Received shutdown signal, sending disconnect message");
                    *state.write().await = RealtimeState::ShuttingDown;

                    // Send goodbye message to frontend before disconnecting
                    let disconnect_msg = RemoteRealtimeMessage::Disconnect {
                        payload: DisconnectPayload {
                            reason: "Bridge shutting down".to_string(),
                            hostname: gethostname::gethostname().to_string_lossy().to_string(),
                        },
                    };
                    if let Err(e) = self.send_broadcast(&mut write, &channel_topic_clone, &user_id, disconnect_msg).await {
                        tracing::warn!("Failed to send disconnect message: {}", e);
                    } else {
                        tracing::info!("Disconnect message sent to frontend");
                    }

                    break;
                }

                // Phoenix protocol heartbeat (keeps connection alive)
                _ = phoenix_heartbeat.tick() => {
                    let hb_ref = self.next_ref().await;
                    let heartbeat_msg = serde_json::json!({
                        "topic": "phoenix",
                        "event": "heartbeat",
                        "payload": {},
                        "ref": hb_ref
                    });
                    let heartbeat_str = match serde_json::to_string(&heartbeat_msg) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("Failed to serialize Phoenix heartbeat: {}", e);
                            break;
                        }
                    };
                    if let Err(e) = write.send(Message::Text(heartbeat_str.into())).await {
                        tracing::error!("Failed to send Phoenix heartbeat: {}", e);
                        break;
                    }
                    tracing::debug!("Sent Phoenix heartbeat");
                }

                // Incoming WebSocket message
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Err(e) = self.handle_incoming_message(
                                &text,
                                &channel_topic_clone,
                                &command_tx,
                                &state,
                            ).await {
                                tracing::error!("Error handling message: {}", e);
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if let Err(e) = write.send(Message::Pong(data)).await {
                                tracing::error!("Failed to send pong: {}", e);
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            tracing::info!("WebSocket closed by server");
                            break;
                        }
                        Some(Err(e)) => {
                            tracing::error!("WebSocket error: {}", e);
                            break;
                        }
                        None => {
                            tracing::info!("WebSocket stream ended");
                            break;
                        }
                        _ => {}
                    }
                }

                // Outgoing message from bridge
                Some(msg) = outgoing_rx.recv() => {
                    if let Err(e) = self.send_broadcast(&mut write, &channel_topic_clone, &user_id, msg).await {
                        tracing::error!("Failed to send broadcast: {}", e);
                    }
                }

                // Heartbeat data from bridge (to broadcast agent status to frontend)
                Some(heartbeat_data) = heartbeat_rx.recv() => {
                    tracing::info!("Broadcasting heartbeat with {} agents to frontend (host: {})",
                        heartbeat_data.agents.len(), heartbeat_data.hostname);
                    let msg = RemoteRealtimeMessage::Heartbeat {
                        payload: HeartbeatPayload {
                            agents: heartbeat_data.agents,
                            system_load: heartbeat_data.system_load,
                            hostname: Some(heartbeat_data.hostname),
                            os: Some(heartbeat_data.os),
                            version: Some(heartbeat_data.version),
                        },
                    };
                    if let Err(e) = self.send_broadcast(&mut write, &channel_topic_clone, &user_id, msg).await {
                        tracing::error!("Failed to send heartbeat broadcast: {}", e);
                    } else {
                        tracing::info!("Heartbeat broadcast sent to channel {}", channel_topic_clone);
                    }
                }

                // Stream data from agent readers (to broadcast to frontend)
                Some((agent_id, chunk_type, content)) = stream_rx.recv() => {
                    tracing::debug!("Broadcasting stream for agent {}: {:?}", agent_id, chunk_type);
                    let msg = RemoteRealtimeMessage::Stream {
                        payload: StreamPayload {
                            agent_id,
                            chunk_type,
                            content,
                        },
                    };
                    if let Err(e) = self.send_broadcast(&mut write, &channel_topic_clone, &user_id, msg).await {
                        tracing::error!("Failed to send stream broadcast: {}", e);
                    }
                }
            }
        }
        *self.state.write().await = RealtimeState::Disconnected;
        Ok(())
    }

    /// Handle incoming WebSocket message
    async fn handle_incoming_message(
        &self,
        text: &str,
        channel_topic: &str,
        command_tx: &mpsc::Sender<BackendCommand>,
        state: &Arc<RwLock<RealtimeState>>,
    ) -> Result<()> {
        let msg: serde_json::Value = serde_json::from_str(text)?;

        let event = msg.get("event").and_then(|e| e.as_str()).unwrap_or("");
        let topic = msg.get("topic").and_then(|t| t.as_str()).unwrap_or("");

        match event {
            "phx_reply" => {
                let status = msg
                    .get("payload")
                    .and_then(|p| p.get("status"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("");

                if status == "ok" && topic == channel_topic {
                    *state.write().await = RealtimeState::Subscribed;
                    tracing::info!("Successfully joined channel: {}", channel_topic);
                } else if status != "ok" {
                    tracing::error!("Join failed: {:?}", msg);
                }
            }

            "broadcast" => {
                tracing::info!("Received broadcast event, full msg: {:?}", msg);

                if let Some(wrapper) = msg.get("payload") {
                    tracing::info!("Broadcast wrapper payload: {:?}", wrapper);

                    if let Some(inner_payload) = wrapper.get("payload") {
                        tracing::info!(
                            "Inner payload (RemoteRealtimeMessage): {:?}",
                            inner_payload
                        );
                        self.handle_remote_message(inner_payload, command_tx)
                            .await?;
                    } else {
                        tracing::warn!("Broadcast has no inner payload: {:?}", wrapper);
                    }
                }
            }

            "presence_state" | "presence_diff" => {
                tracing::debug!("Presence update: {}", event);
            }

            _ => {
                tracing::debug!("Unhandled event: {}", event);
            }
        }

        Ok(())
    }

    /// Handle remote message from server
    async fn handle_remote_message(
        &self,
        msg: &serde_json::Value,
        command_tx: &mpsc::Sender<BackendCommand>,
    ) -> Result<()> {
        let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        tracing::debug!("handle_remote_message: type={}, msg={:?}", msg_type, msg);

        match msg_type {
            "remote.command" => {
                tracing::info!("Received remote.command from frontend");
                if let Some(payload) = msg.get("payload") {
                    tracing::debug!("Command payload: {:?}", payload);
                    match serde_json::from_value::<CommandPayload>(payload.clone()) {
                        Ok(cmd_payload) => {
                            tracing::info!(
                                "Parsed command: type={}, agent_id={:?}",
                                cmd_payload.command_type,
                                cmd_payload.agent_id
                            );
                            let backend_cmd = self.convert_to_backend_command(&cmd_payload)?;
                            command_tx.send(backend_cmd).await?;
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to parse CommandPayload: {}, payload was: {:?}",
                                e,
                                payload
                            );
                        }
                    }
                } else {
                    tracing::warn!("remote.command has no payload");
                }
            }

            "remote.ping" => {
                if let Some(tx) = &self.outgoing_tx {
                    let timestamp = msg
                        .get("payload")
                        .and_then(|p| p.get("serverTimestamp"))
                        .and_then(|t| t.as_i64())
                        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());

                    tx.send(RemoteRealtimeMessage::Pong {
                        payload: PingPongPayload {
                            server_timestamp: timestamp,
                        },
                    })
                    .await?;
                }
            }

            _ => {
                tracing::debug!("Unhandled remote message type: {}", msg_type);
            }
        }

        Ok(())
    }

    /// Convert CommandPayload to BackendCommand
    fn convert_to_backend_command(&self, payload: &CommandPayload) -> Result<BackendCommand> {
        let cmd = match payload.command_type.as_str() {
            "subscribe" => BackendCommand::Subscribe {
                agent_id: payload.agent_id.clone().unwrap_or_default(),
            },
            "unsubscribe" => BackendCommand::Unsubscribe {
                agent_id: payload.agent_id.clone().unwrap_or_default(),
            },
            "send_input" => BackendCommand::SendInput {
                command_id: payload.command_id.clone(),
                agent_id: payload.agent_id.clone().unwrap_or_default(),
                content: payload.content.clone().unwrap_or_default(),
            },
            "slash_command" => BackendCommand::SlashCommand {
                command_id: payload.command_id.clone(),
                agent_id: payload.agent_id.clone().unwrap_or_default(),
                command: payload.command.clone().unwrap_or_default(),
                args: payload.args.clone().unwrap_or_default(),
            },
            "cancel_operation" => BackendCommand::CancelOperation {
                command_id: payload.command_id.clone(),
                agent_id: payload.agent_id.clone().unwrap_or_default(),
            },
            "spawn_agent" => BackendCommand::SpawnAgent {
                command_id: payload.command_id.clone(),
                model: payload.model.clone(),
                working_directory: payload.working_directory.clone(),
            },
            "request_sync" => BackendCommand::RequestSync,
            "ping" => BackendCommand::Ping {
                timestamp: chrono::Utc::now().timestamp_millis(),
            },
            "disconnect" => BackendCommand::Disconnect {
                reason: payload
                    .reason
                    .clone()
                    .unwrap_or_else(|| "Server requested".to_string()),
            },
            _ => bail!("Unknown command type: {}", payload.command_type),
        };

        Ok(cmd)
    }

    /// Send a broadcast message on the channel
    async fn send_broadcast<W>(
        &self,
        write: &mut W,
        channel_topic: &str,
        user_id: &str,
        msg: RemoteRealtimeMessage,
    ) -> Result<()>
    where
        W: SinkExt<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    {
        let msg_ref = self.next_ref().await;

        // Wrap in Realtime broadcast format
        let broadcast = serde_json::json!({
            "topic": channel_topic,
            "event": "broadcast",
            "payload": {
                "type": "broadcast",
                "event": "remote",
                "payload": {
                    "type": get_message_type(&msg),
                    "id": uuid::Uuid::new_v4().to_string(),
                    "payload": get_message_payload(&msg),
                    "timestamp": chrono::Utc::now().timestamp_millis(),
                    "userId": user_id
                }
            },
            "ref": msg_ref
        });

        write
            .send(Message::Text(serde_json::to_string(&broadcast)?.into()))
            .await
            .context("Failed to send broadcast")?;

        Ok(())
    }

    /// Send a message via the outgoing channel (for bridge use)
    pub async fn send(&self, msg: RemoteRealtimeMessage) -> Result<()> {
        if let Some(tx) = &self.outgoing_tx {
            tx.send(msg).await?;
        } else {
            bail!("Not connected");
        }
        Ok(())
    }

    /// Send heartbeat with agent status
    pub async fn send_heartbeat(
        &self,
        heartbeat_data: super::heartbeat::HeartbeatData,
    ) -> Result<()> {
        self.send(RemoteRealtimeMessage::Heartbeat {
            payload: HeartbeatPayload {
                agents: heartbeat_data.agents,
                system_load: heartbeat_data.system_load,
                hostname: Some(heartbeat_data.hostname),
                os: Some(heartbeat_data.os),
                version: Some(heartbeat_data.version),
            },
        })
        .await
    }

    /// Send stream chunk
    pub async fn send_stream(
        &self,
        agent_id: String,
        chunk_type: StreamChunkType,
        content: String,
    ) -> Result<()> {
        self.send(RemoteRealtimeMessage::Stream {
            payload: StreamPayload {
                agent_id,
                chunk_type,
                content,
            },
        })
        .await
    }

    /// Send command result
    pub async fn send_command_result(
        &self,
        command_id: String,
        success: bool,
        result: Option<serde_json::Value>,
        error: Option<String>,
    ) -> Result<()> {
        self.send(RemoteRealtimeMessage::CommandResult {
            payload: CommandResultPayload {
                command_id,
                success,
                result,
                error,
            },
        })
        .await
    }
}

/// Get the message type string for a RemoteRealtimeMessage
fn get_message_type(msg: &RemoteRealtimeMessage) -> &'static str {
    match msg {
        RemoteRealtimeMessage::Register { .. } => "remote.register",
        RemoteRealtimeMessage::Heartbeat { .. } => "remote.heartbeat",
        RemoteRealtimeMessage::Stream { .. } => "remote.stream",
        RemoteRealtimeMessage::CommandResult { .. } => "remote.command_result",
        RemoteRealtimeMessage::Event { .. } => "remote.event",
        RemoteRealtimeMessage::Command { .. } => "remote.command",
        RemoteRealtimeMessage::Ping { .. } => "remote.ping",
        RemoteRealtimeMessage::Pong { .. } => "remote.pong",
        RemoteRealtimeMessage::Disconnect { .. } => "remote.disconnect",
    }
}

/// Get the payload from a RemoteRealtimeMessage
fn get_message_payload(msg: &RemoteRealtimeMessage) -> serde_json::Value {
    match msg {
        RemoteRealtimeMessage::Register { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::Heartbeat { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::Stream { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::CommandResult { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::Event { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::Command { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::Ping { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::Pong { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
        RemoteRealtimeMessage::Disconnect { payload } => {
            serde_json::to_value(payload).unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_config() {
        let config = RealtimeConfig {
            realtime_url: "wss://example.supabase.co/realtime/v1/websocket".to_string(),
            realtime_token: "test_token".to_string(),
            channel_name: "cli:user123".to_string(),
            user_id: "user123".to_string(),
            session_token: "session123".to_string(),
            supabase_anon_key: "test_anon_key".to_string(),
            heartbeat_interval_secs: 30,
            sessions_dir: PathBuf::from("/tmp/test-sessions"),
            version: "0.7.0".to_string(),
        };

        assert_eq!(config.channel_name, "cli:user123");
        assert_eq!(config.version, "0.7.0");
    }

    #[test]
    fn test_message_type() {
        let msg = RemoteRealtimeMessage::Heartbeat {
            payload: HeartbeatPayload {
                agents: vec![],
                system_load: 0.5,
                hostname: Some("test-host".to_string()),
                os: Some("linux".to_string()),
                version: Some("0.1.0".to_string()),
            },
        };

        assert_eq!(get_message_type(&msg), "remote.heartbeat");
    }
}
